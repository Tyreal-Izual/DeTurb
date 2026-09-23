"""Evaluate complete paired videos with reference-output or clip-output DeTurb."""

from __future__ import annotations

import argparse
from deturb.utils.paths import parse_path_args
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from deturb.utils.dynamic_inference import probe_video
from deturb.utils.experiment import collect_runtime_metadata, write_json
from deturb.utils.inference import InferenceStats, decoded_frames, iter_restored_frames, load_model, resolve_device
from deturb.utils.manifest import load_manifest_names
from deturb.utils.metrics import METRIC_POLICY, batch_image_psnr_ssim


def aggregate_video_results(results, aggregation="frame"):
    """Average per-frame scores, weighting each video by its scored frame count."""
    if aggregation not in ("frame", "video"):
        raise ValueError("aggregation must be frame or video")
    if not results or any(r["frames"] <= 0 for r in results):
        raise ValueError("aggregation requires videos with positive scored frame counts")
    frames = sum(r["frames"] for r in results)
    weights = [r["frames"] if aggregation == "frame" else 1 for r in results]
    denominator = sum(weights)
    return {
        "videos": len(results), "frames": frames,
        **{key: math.fsum(r[key] * weight for r, weight in zip(results, weights)) / denominator
           for key in ("psnr", "ssim", "input_psnr", "input_ssim")},
        "perfect_frames": sum(r["perfect_frames"] for r in results),
        "input_perfect_frames": sum(r["input_perfect_frames"] for r in results),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregation", choices=("frame", "video"), default="frame",
                        help="frame: equal weight per scored frame (default); video: equal weight per video")
    args = parse_path_args(parser, argv)
    output_path = Path(args.output).expanduser()
    protected_paths = {
        Path(args.config).expanduser().resolve(), Path(args.checkpoint).expanduser().resolve(),
    }
    if args.manifest_path:
        protected_paths.add(Path(args.manifest_path).expanduser().resolve())
    output_resolved = output_path.resolve()
    if output_resolved in protected_paths:
        parser.error("evaluation output must differ from config/checkpoint/manifest")
    if args.max_videos < 0 or args.max_frames < 0:
        parser.error("limits must be non-negative; zero means the complete set/video")
    root = Path(args.data_root).expanduser()
    names = load_manifest_names(args.manifest_path, "test") if args.manifest_path else sorted(
        p.name for p in (root / "gt").iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if args.max_videos:
        names = names[:args.max_videos]
    if not names:
        raise ValueError("No paired test videos were selected")
    for name in names:
        for kind in ("gt", "turb"):
            if not (root / kind / name).is_file():
                raise FileNotFoundError(root / kind / name)
            if output_resolved == (root / kind / name).resolve():
                parser.error("evaluation output would overwrite a selected input video")
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.config, args.checkpoint, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    results = []
    started = time.perf_counter()
    for name in names:
        source_path, target_path = root / "turb" / name, root / "gt" / name
        source_count, _, height, width = probe_video(source_path)
        target_count, _, target_h, target_w = probe_video(target_path)
        if (source_count, height, width) != (target_count, target_h, target_w):
            raise ValueError(f"Paired video dimensions/frame counts differ: {name}")
        count = min(source_count, args.max_frames or source_count)
        source = decoded_frames(source_path, count)
        targets = decoded_frames(target_path, count)
        stats = InferenceStats()
        totals = dict(psnr=0., ssim=0., input_psnr=0., input_ssim=0., perfect_frames=0, input_perfect_frames=0)
        video_started = time.perf_counter()
        try:
            for frame in iter_restored_frames(source, count, model, model.contract, device,
                                              patch_size=args.patch_size, overlap=args.overlap, stats=stats):
                gt = next(targets)
                target = torch.from_numpy(np.ascontiguousarray(gt[..., ::-1])).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
                output_metrics = batch_image_psnr_ssim(frame.restored.unsqueeze(0).to(device), target)
                input_metrics = batch_image_psnr_ssim(frame.source.unsqueeze(0).to(device).float() / 255, target)
                totals["psnr"] += output_metrics.psnr_sum
                totals["ssim"] += output_metrics.ssim_sum
                totals["input_psnr"] += input_metrics.psnr_sum
                totals["input_ssim"] += input_metrics.ssim_sum
                totals["perfect_frames"] += output_metrics.perfect_frames
                totals["input_perfect_frames"] += input_metrics.perfect_frames
        finally:
            source.close()
            targets.close()
        results.append({"video": name, "frames": count,
                        **{k: v / count if k in ("psnr", "ssim", "input_psnr", "input_ssim") else v
                           for k, v in totals.items()},
                        "pipeline_seconds": time.perf_counter() - video_started,
                        "inference": asdict(stats)})
    elapsed = time.perf_counter() - started
    aggregate = aggregate_video_results(results, args.aggregation)
    frame_count = aggregate["frames"]
    summary = {
        "schema_version": 2, "status": "completed", "model_id": model.contract.model_id,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_iteration": checkpoint["checkpoint_iteration"], "model": model.model_spec(),
        "runtime": collect_runtime_metadata(Path(__file__).resolve().parents[1]),
        "metric_policy": {**METRIC_POLICY, "video_aggregation": (
            "frame_weighted" if args.aggregation == "frame" else "equal_video_mean"
        )},
        "scope": {"data_root": str(root), "max_frames": args.max_frames, "max_videos": args.max_videos,
                  "patch_size": args.patch_size, "overlap": args.overlap, "full_spatial_resolution": True},
        "aggregate": aggregate,
        "pipeline_seconds": elapsed, "pipeline_fps": frame_count / elapsed,
        "timing_note": "excludes model load; includes first/cold tile, decode, tiling and metrics; no encoding",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "per_video": results,
    }
    write_json(output_path, summary)
    print(f"Evaluated {len(results)} videos / {frame_count} frames: {args.output}")


if __name__ == "__main__":
    main()
