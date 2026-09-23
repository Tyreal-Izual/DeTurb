"""Train DeTurb on paired video or Static frame sequences."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from deturb.data.dataset_video_train import DataLoaderTurbVideo
from deturb.model import (
    DeTurb,
    DeTurbLoss,
    load_contract_config,
    load_loss_config,
)
from deturb.utils.checkpoints import (
    capture_rng_state,
    load_training_checkpoint,
    save_training_checkpoint,
)
from deturb.utils.config import parse_args_with_config
from deturb.utils.data_loading import worker_options
from deturb.utils.indexed_data import IndexedAugmentationDataset, IndexedBatchSampler, paired_dataset_fingerprint
from deturb.utils.distributed import (
    DistributedEvalSampler,
    close_distributed_if_initialized,
    initialize_distributed,
)
from deturb.utils.experiment import (
    collect_runtime_metadata,
    configure_logging,
    update_json,
    write_json,
)
from deturb.utils.general import find_latest_checkpoint, get_cuda_info
from deturb.utils.metrics import METRIC_POLICY, batch_model_output_psnr_ssim
from deturb.utils.numerics import raise_if_any_rank_failed, require_finite, tensors_in
from deturb.utils.scheduler import GradualWarmupScheduler
from deturb.utils.tensor_image import tensor_to_uint8_image
from deturb.utils.timing import (
    aggregate_rank_events,
    aggregate_rank_step_timings,
)
from deturb.utils.training import train_step, validation_step


_ACTIVE_EXPERIMENT_PATH: str | None = None


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iters", type=int, default=400000)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data-layout", choices=("triplet", "paired_turb", "static_frames"), default="triplet")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--data-order", dest="data_order", choices=("indexed", "legacy"), default="indexed",
                        help="indexed replays sampling/augmentation exactly; legacy reproduces historical loader behavior")
    parser.add_argument("--video-read-attempts", type=int, default=3)
    parser.add_argument("--video-timeout-ms", type=int, default=10000)
    parser.add_argument("--video-retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=900)
    parser.add_argument("--prefetch-factor", dest="prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--learning-rate", "-l", dest="lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup-iters", dest="warmup_iters", type=int, default=10000)
    parser.add_argument("--log-frequency", dest="log_frequency", type=int, default=100)
    parser.add_argument("--metric-frequency", dest="metric_frequency", type=int, default=100)
    parser.add_argument("--image-frequency", dest="image_frequency", type=int, default=500)
    parser.add_argument("--val-preview-frequency", dest="val_preview_frequency", type=int, default=250)
    parser.add_argument("--print-period", dest="print_period", type=int, default=1000)
    parser.add_argument("--val-period", dest="val_period", type=int, default=5000)
    parser.add_argument(
        "--validation-repeats",
        dest="validation_repeats",
        type=int,
        default=1,
        help="repeat each validation on the unchanged model to verify determinism",
    )
    parser.add_argument(
        "--validate-at-end",
        dest="validate_at_end",
        action="store_true",
        help="run validation at the graceful stop even when it is off cadence",
    )
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="deturb")
    parser.add_argument("--task", choices=("turb", "blur"), default="turb")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", dest="amp_dtype", choices=("none",), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-path", dest="manifest_path", type=str, default=None)
    parser.add_argument("--max-train-samples", dest="max_train_samples", type=int, default=0)
    parser.add_argument("--max-val-samples", dest="max_val_samples", type=int, default=0)
    parser.add_argument("--stop-after-iters", dest="stop_after_iters", type=int, default=0)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--start-over", "--finetune", dest="start_over", action="store_true",
                        help="load compatible weights but reset optimizer, schedule, iteration and best metric")
    parser.add_argument(
        "--expected-start-iteration",
        dest="expected_start_iteration",
        type=int,
        default=None,
        help="fail rather than resume from an unexpected checkpoint iteration",
    )
    parser.add_argument(
        "--timing-warmup-iters",
        dest="timing_warmup_iters",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--timing-measure-iters",
        dest="timing_measure_iters",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--benchmark-force-edge-loss",
        dest="benchmark_force_edge_loss",
        action="store_true",
        help="benchmark-only: activate the configured edge-loss path immediately",
    )
    return parse_args_with_config(parser, "train", argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.data_layout not in ("triplet", "paired_turb", "static_frames"):
        raise ValueError("Invalid data_layout")
    if args.data_layout in ("paired_turb", "static_frames") and args.task != "turb":
        raise ValueError("paired/static data provides no blur input; use task=turb")
    if args.dataset_name:
        if not args.manifest_path:
            raise ValueError("An identified dataset requires a manifest")
        manifest = json.loads(Path(args.manifest_path).read_text())
        if manifest.get("dataset_name") != args.dataset_name:
            raise ValueError("Training manifest dataset identity differs")
        dataset_root = Path(manifest["dataset_root"]).resolve()
        expected_train, expected_test = dataset_root / "train", dataset_root / "test"
        if args.data_layout == "static_frames":
            if manifest.get("data_layout") != "static_frames":
                raise ValueError("Static manifest layout differs")
            expected_train = Path(manifest['split_roots']['train']).resolve()
            expected_test = Path(manifest['split_roots']['test']).resolve()
        if (Path(args.train_path).resolve() != expected_train
                or Path(args.val_path).resolve() != expected_test):
            raise ValueError("Training/validation roots differ from the identified dataset")
    if args.data_order not in ("indexed", "legacy"):
        raise ValueError("data_order must be indexed or legacy")
    if args.start_over and not args.load:
        raise ValueError("--finetune/--start-over requires --load")
    for name in ("train_path", "val_path", "log_path"):
        if not getattr(args, name):
            raise ValueError(f"--{name} is required through config or CLI")
    for name in (
        "iters",
        "batch_size",
        "patch_size",
        "num_frames",
        "log_frequency",
        "metric_frequency",
        "print_period",
        "val_period",
        "validation_repeats",
        "video_read_attempts",
        "video_timeout_ms",
        "ddp_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "num_workers",
        "warmup_iters",
        "image_frequency",
        "val_preview_frequency",
        "max_train_samples",
        "max_val_samples",
        "stop_after_iters",
        "timing_warmup_iters",
        "timing_measure_iters",
        "video_retry_delay_seconds",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.lr <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.patch_size % 32:
        raise ValueError("--patch-size must be divisible by 32")
    if args.amp_dtype != "none":
        raise ValueError("DeTurb currently validates FP32 training only")
    if args.timing_warmup_iters and not args.timing_measure_iters:
        raise ValueError("--timing-warmup-iters requires --timing-measure-iters")
    if args.benchmark_force_edge_loss and not args.timing_measure_iters:
        raise ValueError(
            "--benchmark-force-edge-loss is only valid with timing measurement"
        )
    if (
        args.expected_start_iteration is not None
        and args.expected_start_iteration < 0
    ):
        raise ValueError("--expected-start-iteration cannot be negative")


def _make_run_directories(run_path: Path) -> tuple[Path, Path]:
    image_path = run_path / "imgs"
    checkpoint_path = run_path / "checkpoints"
    image_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path.mkdir(parents=True, exist_ok=False)
    return image_path, checkpoint_path


def _run_training(argv: list[str] | None = None) -> None:
    global _ACTIVE_EXPERIMENT_PATH
    args = get_args(argv)
    validate_args(args)
    contract = load_contract_config(args.config)
    loss_config = load_loss_config(args.config)
    if args.num_frames != contract.input_frames:
        raise ValueError(
            f"train num_frames={args.num_frames} does not match contract "
            f"input_frames={contract.input_frames}"
        )

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if torch.device(device_name).type not in ("cuda", "cpu"):
        raise ValueError("DeTurb supports CPU/CUDA; use --device cpu on a Mac without CUDA")
    context = initialize_distributed(device_name, timeout_seconds=args.ddp_timeout_seconds)
    device = context.device
    pin_memory = device.type == "cuda"
    rank_seed = args.seed + context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(rank_seed)

    run_path: str | None = None
    if context.is_main:
        log_root = Path(args.log_path).expanduser()
        log_root.mkdir(parents=True, exist_ok=True)
        run_path = str(
            log_root
            / f"{args.run_name}_{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}"
        )
        image_path, checkpoint_path = _make_run_directories(Path(run_path))
    run_path = context.broadcast_object(run_path)
    if run_path is None:
        raise RuntimeError("rank 0 did not provide a run path")
    image_path = Path(run_path) / "imgs"
    checkpoint_path = Path(run_path) / "checkpoints"
    context.barrier()

    configure_logging(Path(run_path) / "recording.log", enabled=context.is_main)
    if context.is_main:
        get_cuda_info(logging)

    dataset_class = DataLoaderTurbVideo
    if args.data_layout == "static_frames":
        from deturb.data.dataset_static_clip import DataLoaderTurbStatic
        dataset_class = DataLoaderTurbStatic
    train_dataset = dataset_class(
        args.train_path,
        num_frames=contract.input_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=True,
        manifest_path=args.manifest_path,
        manifest_split="train" if args.manifest_path else None,
        read_attempts=args.video_read_attempts,
        video_timeout_ms=args.video_timeout_ms,
        retry_delay_seconds=args.video_retry_delay_seconds,
        data_layout=args.data_layout,
    )
    val_dataset = dataset_class(
        args.val_path,
        num_frames=contract.input_frames,
        patch_size=args.patch_size,
        noise=None,
        is_train=False,
        manifest_path=args.manifest_path,
        manifest_split="test" if args.manifest_path else None,
        read_attempts=args.video_read_attempts,
        video_timeout_ms=args.video_timeout_ms,
        retry_delay_seconds=args.video_retry_delay_seconds,
        data_layout=args.data_layout,
    )
    if args.max_train_samples:
        train_dataset = Subset(
            train_dataset,
            range(min(args.max_train_samples, len(train_dataset))),
        )
    if args.max_val_samples:
        val_dataset = Subset(
            val_dataset,
            range(min(args.max_val_samples, len(val_dataset))),
        )

    train_sampler = None
    indexed_sampler = None
    data_spec = None
    data_fingerprint = None
    val_sampler = None
    if context.distributed:
        if args.data_order == "legacy":
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=context.world_size, rank=context.rank,
                shuffle=True, seed=args.seed, drop_last=True,
            )
        val_sampler = DistributedEvalSampler(
            val_dataset,
            rank=context.rank,
            world_size=context.world_size,
        )
    loader_options = worker_options(
        args.num_workers,
        args.persistent_workers,
        args.prefetch_factor,
    )
    if args.data_order == "indexed":
        data_fingerprint = paired_dataset_fingerprint(train_dataset)
        indexed_sampler = IndexedBatchSampler(len(train_dataset), args.batch_size, seed=args.seed,
                                              rank=context.rank, world_size=context.world_size)
        data_spec = indexed_sampler.spec(data_fingerprint)
        if any(spec != data_spec for spec in context.gather_objects(data_spec)):
            raise ValueError("Training ranks disagree on the indexed dataset/order")
        train_dataset = IndexedAugmentationDataset(train_dataset)
        train_loader = DataLoader(train_dataset, batch_sampler=indexed_sampler, num_workers=args.num_workers,
                                  pin_memory=pin_memory, generator=torch.Generator().manual_seed(rank_seed),
                                  **loader_options)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
            sampler=train_sampler, num_workers=args.num_workers, drop_last=True,
            pin_memory=pin_memory, generator=torch.Generator().manual_seed(rank_seed), **loader_options,
        )
    batches_per_epoch = len(train_loader)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(rank_seed + 100000),
        **loader_options,
    )
    if len(train_loader) == 0 or len(val_dataset) == 0:
        raise ValueError("training and validation loaders must be non-empty")

    model = DeTurb.from_config(args.config).to(device)
    model_spec = model.model_spec()
    checkpoint_version = max(model.checkpoint_version, 7 if indexed_sampler is not None else 0)
    if context.is_main:
        logging.info(
            "Model %s: %d input frames -> %d output frames; mode=%s; checkpoint schema=%d",
            contract.model_id, contract.input_frames, contract.output_frames,
            contract.output_mode, checkpoint_version,
        )
    if context.distributed:
        ddp_kwargs: dict[str, object] = {"broadcast_buffers": False}
        if device.type == "cuda":
            ddp_kwargs.update(
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        model = DistributedDataParallel(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    total_iters = args.iters
    warmup_iters = min(args.warmup_iters, max(total_iters - 1, 0))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_iters - warmup_iters, 1),
        eta_min=1e-6,
    )
    scheduler = (
        GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=warmup_iters,
            after_scheduler=cosine,
        )
        if warmup_iters
        else cosine
    )
    criterion = DeTurbLoss(loss_config).to(device)

    start_iter = 0
    best_psnr = 0.0
    loaded_checkpoint: str | None = None
    if args.load:
        if args.load == "latest":
            load_path = find_latest_checkpoint(args.log_path, args.run_name)
            if not load_path:
                raise FileNotFoundError("no latest DeTurb checkpoint found")
        else:
            load_path = args.load
        loaded_checkpoint = str(Path(load_path).expanduser().resolve())
        state = load_training_checkpoint(
            loaded_checkpoint,
            model,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_training=not args.start_over,
            rng_rank=context.rank,
            expected_model_name=contract.model_id,
            minimum_checkpoint_version=5 if args.start_over else checkpoint_version,
            expected_model_spec=model_spec,
            expected_training_config=vars(args),
            expected_loss_config=loss_config.as_metadata(),
            expected_world_size=context.world_size,
            expected_data_spec=data_spec,
            allow_mask_upgrade=args.start_over,
        )
        if not args.start_over:
            start_iter = state["iteration"]
            best_psnr = state["best_psnr"]
    if (
        args.expected_start_iteration is not None
        and start_iter != args.expected_start_iteration
    ):
        raise ValueError(
            f"expected start iteration {args.expected_start_iteration}, "
            f"loaded {start_iter}"
        )

    runtime = collect_runtime_metadata(
        project_root=Path(__file__).resolve().parents[1]
    )
    resolved = {
        "train": vars(args),
        "model": model_spec,
        "loss": loss_config.as_metadata(),
        "distributed_world_size": context.world_size,
        "global_batch_size": args.batch_size * context.world_size,
        "output_frames_per_clip": contract.output_frames,
        "metric_scope": "all_clip_frames" if contract.returns_full_clip else "reference_frame",
        "metric_policy": METRIC_POLICY,
        "data_pipeline": data_spec if data_spec is not None else {"policy": "legacy"},
        "resume": {
            "requested": args.load,
            "checkpoint_path": loaded_checkpoint,
            "resume_training": bool(args.load and not args.start_over),
            "start_iteration": start_iter,
            "start_best_psnr": best_psnr,
        },
    }
    experiment_path = str(Path(run_path) / "experiment.json")
    if context.is_main:
        write_json(Path(run_path) / "resolved_config.json", resolved)
        write_json(
            experiment_path,
            {
                "status": "running",
                "command": [sys.executable, *sys.argv],
                "runtime": runtime,
                "resolved": resolved,
                "datasets": {
                    "train_path": args.train_path,
                    "train_samples": len(train_dataset),
                    "val_path": args.val_path,
                    "val_samples": len(val_dataset),
                },
            },
        )
        _ACTIVE_EXPERIMENT_PATH = experiment_path

    run_end_iter = (
        min(total_iters, args.stop_after_iters)
        if args.stop_after_iters
        else total_iters
    )
    required_timing_steps = args.timing_warmup_iters + args.timing_measure_iters
    if required_timing_steps > run_end_iter - start_iter:
        raise ValueError(
            "timing warm-up and measurement exceed this invocation's iterations"
        )
    input_index = 0 if args.task == "blur" else 1
    iteration = start_iter
    epoch, start_batch = divmod(start_iter, batches_per_epoch)
    last_iteration_end = time.perf_counter()
    loss_sums = {
        key: 0.0
        for key in (
            "total",
            "final",
            "alignment_full",
            "alignment_mid",
            "alignment_coarse",
            "edge",
        )
    }
    steps_since_log = 0
    data_wait_seconds = 0.0
    model_step_seconds = 0.0
    metric_psnr_sum = 0.0
    metric_ssim_sum = 0.0
    metric_count = 0
    metric_perfect_frames = 0
    validation_runs = 0
    last_validation: dict[str, object] | None = None
    validation_history: list[dict[str, object]] = []
    training_history: list[dict[str, object]] = []
    local_step_timings: list[dict[str, float]] = []
    local_checkpoint_events: list[dict[str, object]] = []
    local_validation_events: list[dict[str, object]] = []
    invocation_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save_checkpoints(filenames: list[str]) -> None:
        require_finite((*model.parameters(), *tensors_in(optimizer.state)),
                       "checkpoint preparation", device, synchronize=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        checkpoint_started = time.perf_counter()
        rng_states = context.gather_objects(capture_rng_state())
        save_error = None
        if context.is_main:
            try:
                for filename in filenames:
                    save_training_checkpoint(
                        checkpoint_path / filename, model, optimizer, scheduler,
                        iteration, best_psnr, resolved, rng_states=rng_states,
                        metadata={"runtime": runtime, "model": model_spec},
                        checkpoint_version=checkpoint_version, model_name=contract.model_id,
                        data_state=(indexed_sampler.state_dict(iteration, data_fingerprint)
                                    if indexed_sampler is not None else None),
                    )
            except Exception as error:
                save_error = f"{type(error).__name__}: {error}"
        save_error = context.broadcast_object(save_error)
        if save_error:
            raise OSError(f"Checkpoint save failed on rank 0: {save_error}")
        local_checkpoint_events.append(
            {
                "iteration": iteration,
                "filenames": list(filenames),
                "seconds": time.perf_counter() - checkpoint_started,
            }
        )

    while iteration < run_end_iter:
        if indexed_sampler is not None:
            indexed_sampler.set_epoch(epoch, start_batch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for data in train_loader:
            iteration_started = last_iteration_end
            data_ready = time.perf_counter()
            current_data_wait = data_ready - iteration_started
            data_wait_seconds += current_data_wait
            input_sequence = data[input_index].to(device, non_blocking=pin_memory)
            target_sequence = data[2].to(device, non_blocking=pin_memory)
            model_started = time.perf_counter()
            result = train_step(
                model,
                optimizer,
                scheduler,
                criterion,
                input_sequence,
                target_sequence,
                contract,
                iteration=(
                    max(iteration, loss_config.edge_start_iteration)
                    if args.benchmark_force_edge_loss
                    else iteration
                ),
            )
            if args.timing_measure_iters and device.type == "cuda":
                torch.cuda.synchronize(device)
            model_finished = time.perf_counter()
            current_model_step = model_finished - model_started
            model_step_seconds += current_model_step
            iteration += 1
            invocation_step += 1
            should_measure_iteration = (
                args.timing_warmup_iters
                < invocation_step
                <= required_timing_steps
            )
            steps_since_log += 1
            for key in loss_sums:
                loss_sums[key] += float(result.losses[key])

            is_final = iteration >= run_end_iter
            if iteration % args.metric_frequency == 0 or is_final:
                metrics = batch_model_output_psnr_ssim(result.restored, result.target)
                metric_psnr_sum += metrics.psnr_sum
                metric_ssim_sum += metrics.ssim_sum
                metric_count += metrics.count
                metric_perfect_frames += metrics.perfect_frames
            if (
                context.is_main
                and args.image_frequency
                and iteration % args.image_frequency == 0
            ):
                reference_input = input_sequence[0, contract.reference_index]
                preview = Image.fromarray(
                    np.concatenate(
                        (
                            tensor_to_uint8_image(reference_input),
                            tensor_to_uint8_image(contract.output_reference(result.restored)[0]),
                            tensor_to_uint8_image(contract.output_reference(result.target)[0]),
                        ),
                        axis=1,
                    )
                ).convert("RGB")
                preview.save(image_path / f"train_{iteration}.jpg", "JPEG")

            if iteration % args.log_frequency == 0 or is_final:
                values = [
                    *(loss_sums[key] for key in loss_sums),
                    steps_since_log,
                    data_wait_seconds,
                    model_step_seconds,
                    metric_psnr_sum,
                    metric_ssim_sum,
                    metric_count,
                    metric_perfect_frames,
                ]
                reduced = context.reduce_sums(values)
                if context.is_main:
                    loss_count = len(loss_sums)
                    reduced_losses = reduced[:loss_count]
                    step_count, wait_sum, model_sum, psnr_sum, ssim_sum, count, perfect_frames = (
                        reduced[loss_count:]
                    )
                    loss_message = " ".join(
                        f"{key}:{value / step_count:.6f}"
                        for key, value in zip(loss_sums, reduced_losses)
                    )
                    metric_message = (
                        f"PSNR:{psnr_sum / count:.3f} SSIM:{ssim_sum / count:.5f}"
                        if count
                        else "PSNR:n/a SSIM:n/a"
                    )
                    training_history.append(
                        {
                            "iteration": iteration,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "losses": {
                                key: value / step_count
                                for key, value in zip(loss_sums, reduced_losses)
                            },
                            "data_wait_seconds": wait_sum / step_count,
                            "model_step_seconds": model_sum / step_count,
                            "psnr": psnr_sum / count if count else None,
                            "ssim": ssim_sum / count if count else None,
                            "metric_count": int(count),
                            "perfect_frames": int(perfect_frames),
                        }
                    )
                    logging.info(
                        "Training iter %d/%d LR:%.8f %s DataWait:%.6f "
                        "ModelStep:%.6f %s",
                        iteration,
                        total_iters,
                        optimizer.param_groups[0]["lr"],
                        loss_message,
                        wait_sum / step_count,
                        model_sum / step_count,
                        metric_message,
                    )
                loss_sums = {key: 0.0 for key in loss_sums}
                steps_since_log = 0
                data_wait_seconds = 0.0
                model_step_seconds = 0.0
                metric_psnr_sum = 0.0
                metric_ssim_sum = 0.0
                metric_count = 0
                metric_perfect_frames = 0

            if iteration % args.print_period == 0 or is_final:
                save_checkpoints([f"model_{iteration}.pth", "latest.pth"])

            if iteration % args.val_period == 0 or (
                is_final and args.validate_at_end
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_started = time.perf_counter()
                repeat_results: list[dict[str, float | int]] = []
                for repeat_index in range(args.validation_repeats):
                    totals = [0.0] * 6
                    validation_failed = False
                    for batch_index, val_data in enumerate(val_loader):
                        val_input = val_data[input_index].to(
                            device,
                            non_blocking=pin_memory,
                        )
                        val_target = val_data[2].to(device, non_blocking=pin_memory)
                        try:
                            validation = validation_step(
                                model, criterion, val_input, val_target, contract,
                                iteration=iteration,
                            )
                        except FloatingPointError:
                            validation_failed = True
                            break
                        metrics = batch_model_output_psnr_ssim(
                            validation.restored,
                            validation.target,
                        )
                        batch_size = validation.target.shape[0]
                        totals[0] += float(validation.losses["total"]) * batch_size
                        totals[1] += batch_size
                        totals[2] += metrics.psnr_sum
                        totals[3] += metrics.ssim_sum
                        totals[4] += metrics.count
                        totals[5] += metrics.perfect_frames
                        if (
                            repeat_index == 0
                            and context.is_main
                            and args.val_preview_frequency
                            and batch_index % args.val_preview_frequency == 0
                        ):
                            preview = Image.fromarray(
                                np.concatenate(
                                    (
                                        tensor_to_uint8_image(
                                            val_input[0, contract.reference_index]
                                        ),
                                        tensor_to_uint8_image(contract.output_reference(validation.restored)[0]),
                                        tensor_to_uint8_image(contract.output_reference(validation.target)[0]),
                                    ),
                                    axis=1,
                                )
                            ).convert("RGB")
                            preview.save(
                                image_path / f"val_{iteration}_{batch_index}.jpg",
                                "JPEG",
                            )
                    # A single collective after the shard avoids deadlock when
                    # ranks have unequal numbers of validation batches.
                    raise_if_any_rank_failed(validation_failed, "distributed validation", device)
                    loss_sum, samples, psnr_sum, ssim_sum, count, perfect_frames = (
                        context.reduce_sums(totals)
                    )
                    if samples <= 0 or count <= 0:
                        raise RuntimeError("distributed validation produced no samples")
                    repeat_results.append(
                        {
                            "iteration": iteration,
                            "loss": loss_sum / samples,
                            "samples": int(samples),
                            "psnr": psnr_sum / count,
                            "ssim": ssim_sum / count,
                            "metric_count": int(count),
                            "perfect_frames": int(perfect_frames),
                        }
                    )
                primary_validation = repeat_results[0]
                repeats_bitwise_equal = all(
                    result == primary_validation for result in repeat_results[1:]
                )
                if not repeats_bitwise_equal:
                    raise RuntimeError(
                        "repeated distributed validation was not bitwise deterministic"
                    )
                validation_runs += 1
                last_validation = {
                    **primary_validation,
                    "repeat_count": args.validation_repeats,
                    "repeats_bitwise_equal": repeats_bitwise_equal,
                    "repeat_results": repeat_results,
                }
                validation_history.append(last_validation)
                psnr = float(primary_validation["psnr"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                local_validation_events.append(
                    {
                        "iteration": iteration,
                        "samples": int(primary_validation["samples"]),
                        "repeat_count": args.validation_repeats,
                        "seconds": time.perf_counter() - validation_started,
                    }
                )
                if context.is_main:
                    logging.info(
                        "Validation iter %d/%d Loss:%.6f PSNR:%.3f SSIM:%.5f",
                        iteration,
                        total_iters,
                        primary_validation["loss"],
                        psnr,
                        primary_validation["ssim"],
                    )
                if psnr > best_psnr:
                    best_psnr = psnr
                    save_checkpoints(["model_best.pth", "latest.pth"])
                model.train()

            iteration_finished = time.perf_counter()
            if should_measure_iteration:
                local_step_timings.append(
                    {
                        "end_to_end_seconds": iteration_finished - iteration_started,
                        "model_step_seconds": current_model_step,
                        "data_wait_seconds": current_data_wait,
                    }
                )
            last_iteration_end = iteration_finished
            if is_final:
                break
        epoch += 1
        start_batch = 0

    if args.timing_measure_iters and len(local_step_timings) != args.timing_measure_iters:
        raise RuntimeError("trainer did not collect every requested timing iteration")

    local_peak: dict[str, int] = {}
    if device.type == "cuda":
        local_peak = {
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    peaks = context.gather_objects(local_peak)
    performance_profile = None
    if args.timing_measure_iters:
        rank_performance = context.gather_objects(
            {
                "rank": context.rank,
                "steps": local_step_timings,
                "checkpoint_events": local_checkpoint_events,
                "validation_events": local_validation_events,
            }
        )
        if context.is_main:
            performance_profile = {
                "training": aggregate_rank_step_timings(
                    rank_performance,
                    warmup_iterations=args.timing_warmup_iters,
                    measured_iterations=args.timing_measure_iters,
                    global_batch_size=args.batch_size * context.world_size,
                    force_edge_loss=args.benchmark_force_edge_loss,
                ),
                "checkpoint_events": aggregate_rank_events(
                    rank_performance,
                    "checkpoint_events",
                ),
                "validation_events": aggregate_rank_events(
                    rank_performance,
                    "validation_events",
                ),
            }
    if context.is_main:
        peak_summary = (
            {key: max(item[key] for item in peaks) for key in local_peak}
            if local_peak
            else {}
        )
        update_json(
            experiment_path,
            {
                "status": "completed",
                "stopped_early": run_end_iter < total_iters,
                "completed_iterations": iteration,
                "best_psnr": best_psnr,
                "completed_at": datetime.now().astimezone().isoformat(),
                "validation_runs": validation_runs,
                "last_validation": last_validation,
                "training_history": training_history,
                "validation_history": validation_history,
                "performance_profile": performance_profile,
                **peak_summary,
            },
        )
        _ACTIVE_EXPERIMENT_PATH = None
    context.close()


def main(argv: list[str] | None = None) -> None:
    global _ACTIVE_EXPERIMENT_PATH
    try:
        _run_training(argv)
    except Exception as error:
        if _ACTIVE_EXPERIMENT_PATH and int(os.environ.get("RANK", "0")) == 0:
            try:
                update_json(
                    _ACTIVE_EXPERIMENT_PATH,
                    {
                        "status": "failed",
                        "failed_at": datetime.now().astimezone().isoformat(),
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception:
                logging.exception("failed to update DeTurb experiment metadata")
        raise
    finally:
        _ACTIVE_EXPERIMENT_PATH = None
        close_distributed_if_initialized()


def entrypoint():
    main()


if __name__ == "__main__":
    main()
