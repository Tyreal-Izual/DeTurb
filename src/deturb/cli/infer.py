"""Restore a complete video with either DeTurb output contract, using bounded windows."""

from __future__ import annotations

import argparse
from deturb.utils.paths import parse_path_args
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from deturb.utils.dynamic_inference import probe_video
from deturb.utils.experiment import write_json
from deturb.utils.inference import InferenceStats, decoded_frames, iter_restored_frames, load_model, resolve_device
from deturb.utils.tensor_image import tensor_to_uint8_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parse_path_args(parser, argv)
    source, destination = Path(args.input).expanduser(), Path(args.output).expanduser()
    if source.resolve() == destination.resolve():
        parser.error("input and output must differ")
    if destination.suffix.lower() != ".mp4":
        parser.error("output must be an .mp4 file")
    if destination.with_suffix(".json").resolve() in {
        source.resolve(), Path(args.config).expanduser().resolve(), Path(args.checkpoint).expanduser().resolve(),
    }:
        parser.error("output metadata path would overwrite an input/config/checkpoint")
    total, fps, height, width = probe_video(source)
    if not 0 <= args.start_frame < total or args.max_frames < 0 or not math.isfinite(fps) or fps <= 0:
        parser.error("invalid frame selection or source FPS")
    count = min(total - args.start_frame, args.max_frames or total)
    device = resolve_device(args.device)
    model, state = load_model(args.config, args.checkpoint, device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.stem}.", suffix=".mp4")
    os.close(fd)
    temporary = Path(temporary_name)
    encoded_h, encoded_w = height + height % 2, width + width % 2
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (encoded_w, encoded_h))
    stats = InferenceStats()
    started = time.perf_counter()
    frames = decoded_frames(source, count, args.start_frame)
    try:
        if not writer.isOpened():
            raise OSError(f"Cannot create output video: {destination}")
        for frame in iter_restored_frames(frames, count, model, model.contract, device,
                                          patch_size=args.patch_size, overlap=args.overlap, stats=stats):
            bgr = cv2.cvtColor(tensor_to_uint8_image(frame.restored), cv2.COLOR_RGB2BGR)
            if (encoded_h, encoded_w) != (height, width):
                bgr = cv2.copyMakeBorder(bgr, 0, encoded_h - height, 0, encoded_w - width, cv2.BORDER_REPLICATE)
            writer.write(bgr)
        writer.release()
        written, _, out_h, out_w = probe_video(temporary)
        if (written, out_h, out_w) != (count, encoded_h, encoded_w):
            raise OSError("Encoded output frame count/dimensions do not match the restored video")
        os.replace(temporary, destination)
    finally:
        frames.close()
        writer.release()
        temporary.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    write_json(destination.with_suffix(".json"), {
        "status": "completed", "model_id": model.contract.model_id,
        "checkpoint_iteration": state["checkpoint_iteration"], "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(source), "output": str(destination), "start_frame": args.start_frame,
        "frames": count, "fps": fps, "source_shape": [height, width],
        "encoded_shape": [encoded_h, encoded_w], "audio_preserved": False,
        "patch_size": args.patch_size, "overlap": args.overlap,
        "pipeline_seconds": elapsed, "pipeline_fps": count / elapsed,
        "inference": asdict(stats), "timing_note": "excludes model load; includes first/cold tile, decoding and encoding",
    })
    print(f"Restored {count} frames: {destination}")


if __name__ == "__main__":
    main()
