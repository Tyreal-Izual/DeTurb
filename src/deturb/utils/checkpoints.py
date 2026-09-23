"""Checkpoint save and resume support for Dynamic DeTurb training."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .numerics import all_finite, state_is_finite, tensors_in
from .indexed_data import validate_data_state


RESUME_TRAIN_FIELDS = (
    "iters", "lr", "weight_decay", "warmup_iters", "batch_size", "patch_size",
    "num_frames", "seed", "task", "amp_dtype", "train_path", "val_path",
    "manifest_path", "max_train_samples", "max_val_samples",
    "data_order", "data_layout", "dataset_name",
)
_SCHEDULER_POLICY_FIELDS = (
    "T_max", "eta_min", "total_epoch", "multiplier", "base_lrs", "step_size",
    "gamma", "milestones", "mode", "factor", "patience", "threshold",
    "threshold_mode", "cooldown", "min_lrs", "eps",
)


def _scheduler_policy(state: dict[str, Any] | None) -> Any:
    if state is None:
        return None
    if not isinstance(state, dict) or not isinstance(state.get("outer"), dict):
        raise ValueError("Checkpoint scheduler state is incomplete")
    if not isinstance(state.get("wrapped"), bool):
        raise ValueError("Checkpoint scheduler wrapper state is incomplete")
    def policy(values):
        if values is None:
            return None
        if not isinstance(values, dict) or not {"last_epoch", "_step_count", "_last_lr"} <= values.keys():
            raise ValueError("Checkpoint scheduler progress state is incomplete")
        return {key: values[key] for key in _SCHEDULER_POLICY_FIELDS if key in values}
    return {"wrapped": state.get("wrapped"), "outer": policy(state["outer"]),
            "after": policy(state.get("after"))}


def validate_resume_state(
    checkpoint: dict[str, Any], *, scheduler: Any = None,
    check_scheduler_policy: bool = False, rng_rank: int = 0,
    expected_world_size: int | None = None,
    require_cuda_rng: bool = False,
) -> None:
    """Validate required resumable state before mutating model/optimizer/RNG."""
    for key in ("iter", "optimizer", "scheduler", "rng_states", "best_psnr"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing required resume state: {key}")
    iteration = checkpoint["iter"]
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("Checkpoint iteration must be a non-negative integer")
    if not math.isfinite(float(checkpoint["best_psnr"])):
        raise ValueError("Checkpoint best_psnr must be finite; use --finetune to reset training state")
    if checkpoint.get("checkpoint_version", 0) >= 7:
        validate_data_state(checkpoint.get("data_state"), iteration)
    optimizer = checkpoint["optimizer"]
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not optimizer.get("param_groups"):
        raise ValueError("Checkpoint optimizer state is incomplete")
    if not state_is_finite(optimizer):
        raise ValueError("Checkpoint optimizer state contains non-finite values")
    states = checkpoint["rng_states"]
    if not isinstance(states, list) or not states or not 0 <= rng_rank < len(states):
        raise ValueError("Checkpoint does not contain the requested rank RNG state")
    if expected_world_size is not None and len(states) != expected_world_size:
        raise ValueError("Checkpoint RNG rank count does not match world size")
    if checkpoint.get("checkpoint_version", 0) >= 7 and checkpoint["data_state"]["world_size"] != len(states):
        raise ValueError("Checkpoint data world size does not match RNG rank count")
    for state in states:
        if not isinstance(state, dict) or not {"python", "numpy", "torch", "cuda"} <= state.keys():
            raise ValueError("Checkpoint rank RNG state is incomplete")
        if state["python"] is None or state["numpy"] is None or not isinstance(state["torch"], torch.Tensor):
            raise ValueError("Checkpoint rank RNG state is invalid")
        try:
            random.Random().setstate(state["python"])
            np.random.RandomState(0).set_state(state["numpy"])
            torch.Generator().set_state(state["torch"].cpu())
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("Checkpoint rank RNG state is invalid") from error
        cuda_states = state["cuda"]
        if require_cuda_rng and not cuda_states:
            raise ValueError("CUDA resume requires saved CUDA RNG state")
        if cuda_states is not None and (
            not isinstance(cuda_states, list) or not cuda_states
            or any(not isinstance(s, torch.Tensor) or s.dtype != torch.uint8 or s.ndim != 1
                   for s in cuda_states)
        ):
            raise ValueError("Checkpoint CUDA RNG state is invalid")
    saved_scheduler = checkpoint["scheduler"]
    if not state_is_finite((saved_scheduler, states)):
        raise ValueError("Checkpoint scheduler/RNG state contains non-finite values")
    _scheduler_policy(saved_scheduler)
    if check_scheduler_policy:
        current_scheduler = capture_scheduler_state(scheduler)
        if _scheduler_policy(saved_scheduler) != _scheduler_policy(current_scheduler):
            raise ValueError("Checkpoint scheduler policy differs; use --finetune for a new training schedule")
        if saved_scheduler is not None:
            for key in ("class", "after_class"):
                if key in saved_scheduler and saved_scheduler[key] != current_scheduler.get(key):
                    raise ValueError("Checkpoint scheduler class differs; use --finetune")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["cuda"]]
        )


def capture_scheduler_state(scheduler: Any) -> dict[str, Any] | None:
    if scheduler is None:
        return None
    state = scheduler.state_dict()
    if hasattr(scheduler, "after_scheduler"):
        outer_state = dict(state)
        outer_state.pop("after_scheduler", None)
        after_scheduler = scheduler.after_scheduler
        return {
            "class": type(scheduler).__qualname__,
            "after_class": type(after_scheduler).__qualname__ if after_scheduler is not None else None,
            "wrapped": True,
            "outer": outer_state,
            "after": (
                after_scheduler.state_dict()
                if after_scheduler is not None
                else None
            ),
        }
    return {"class": type(scheduler).__qualname__, "wrapped": False, "outer": state, "after": None}


def restore_scheduler_state(scheduler: Any, state: dict[str, Any] | None) -> None:
    if scheduler is None or state is None:
        return
    scheduler.load_state_dict(state["outer"])
    if state.get("wrapped") and state.get("after") is not None:
        scheduler.after_scheduler.load_state_dict(state["after"])


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    iteration: int,
    best_psnr: float,
    config: dict[str, Any],
    rng_states: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    checkpoint_version: int = 4,
    model_name: str = "legacy_dynamic_v1",
    data_state: dict[str, Any] | None = None,
) -> None:
    if checkpoint_version <= 0:
        raise ValueError("checkpoint_version must be positive")
    if not model_name:
        raise ValueError("model_name cannot be empty")
    if checkpoint_version >= 7:
        validate_data_state(data_state, iteration)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "checkpoint_version": checkpoint_version,
            "model_name": model_name,
            "iter": iteration,
            "best_psnr": best_psnr,
            "state_dict": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": capture_scheduler_state(scheduler),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng_states": rng_states or [capture_rng_state()],
            "config": config,
            "metadata": metadata or {},
            "data_state": data_state,
        }
    if not state_is_finite(payload):
        raise FloatingPointError("Refusing to save a checkpoint with non-finite model/optimizer/metric state")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=checkpoint_path.parent, prefix=f".{checkpoint_path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    resume_training: bool = True,
    rng_rank: int = 0,
    scaler: torch.amp.GradScaler | None = None,
    expected_model_name: str | None = None,
    minimum_checkpoint_version: int | None = None,
    expected_model_spec: dict[str, Any] | None = None,
    expected_training_config: dict[str, Any] | None = None,
    expected_loss_config: dict[str, Any] | None = None,
    expected_world_size: int | None = None,
    expected_data_spec: dict[str, Any] | None = None,
    allow_mask_upgrade: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(path).expanduser(),
        map_location=device,
        weights_only=False,
    )
    if expected_model_name is not None:
        actual_model_name = checkpoint.get("model_name")
        if actual_model_name != expected_model_name:
            raise ValueError(
                f"Checkpoint model_name is {actual_model_name!r}; "
                f"expected {expected_model_name!r}"
            )
    if minimum_checkpoint_version is not None:
        actual_version = int(checkpoint.get("checkpoint_version", 0))
        if actual_version < minimum_checkpoint_version:
            raise ValueError(
                f"Checkpoint version {actual_version} is older than required "
                f"version {minimum_checkpoint_version}"
            )
    if expected_model_spec is not None:
        saved_spec = checkpoint.get("config", {}).get("model", {})
        for section in ("contract", "registration", "fusion"):
            saved = dict(saved_spec.get(section, {}))
            expected = dict(expected_model_spec.get(section, {}))
            if section == "registration":
                # Schema-5 reference checkpoints predate this optional layer.
                saved.setdefault("post_warp_kernel", None)
                expected.setdefault("post_warp_kernel", None)
                saved.setdefault("upsample_mode", "transpose")
                expected.setdefault("upsample_mode", "transpose")
            if section == "fusion":
                saved.setdefault("attention_mask_mode", "legacy")
                expected.setdefault("attention_mask_mode", "legacy")
                if (allow_mask_upgrade and not resume_training and saved["attention_mask_mode"] == "legacy"
                        and expected["attention_mask_mode"] == "strict"):
                    saved["attention_mask_mode"] = "strict"
            # Configs loaded from JSON contain lists, while constructors may
            # supply tuples. Compare their serialized architectural meaning.
            if json.dumps(saved, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise ValueError(f"Checkpoint model {section} config does not match the requested model")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not all_finite(tensors_in(state_dict)):
        raise ValueError("Checkpoint model weights contain non-finite values")
    if resume_training:
        if optimizer is None or "optimizer" not in checkpoint:
            raise ValueError("Checkpoint does not contain resumable optimizer state")
        if int(checkpoint.get("checkpoint_version", 0)) >= 5:
            validate_resume_state(checkpoint, scheduler=scheduler, check_scheduler_policy=True,
                                  rng_rank=rng_rank, expected_world_size=expected_world_size,
                                  require_cuda_rng=device.type == "cuda")
        if expected_data_spec is not None:
            validate_data_state(checkpoint.get("data_state"), checkpoint.get("iter", 0), expected_data_spec)
        saved_config = checkpoint.get("config", {})
        if expected_training_config is not None:
            saved_train = saved_config.get("train", {})
            historical_defaults = {"data_order": "legacy", "data_layout": "triplet"}
            differences = [key for key in RESUME_TRAIN_FIELDS
                           if saved_train.get(key, historical_defaults.get(key))
                           != expected_training_config.get(key, historical_defaults.get(key))]
            if differences:
                raise ValueError("Resume training config differs for " + ", ".join(differences)
                                 + "; use --finetune to reset optimizer/scheduler/iteration")
        if expected_loss_config is not None and json.dumps(saved_config.get("loss"), sort_keys=True) != json.dumps(expected_loss_config, sort_keys=True):
            raise ValueError("Resume loss config differs; use --finetune to change the objective")
        saved_groups = checkpoint["optimizer"].get("param_groups", [])
        if len(saved_groups) != len(optimizer.param_groups):
            raise ValueError("Checkpoint optimizer parameter groups differ")
        for saved_group, current_group in zip(saved_groups, optimizer.param_groups):
            if len(saved_group.get("params", [])) != len(current_group["params"]):
                raise ValueError("Checkpoint optimizer parameter group sizes differ")
            for key in ("betas", "eps", "weight_decay", "amsgrad", "maximize", "momentum", "dampening", "nesterov"):
                if saved_group.get(key) != current_group.get(key):
                    raise ValueError(f"Checkpoint optimizer policy differs for {key}; use --finetune")
    unwrap_model(model).load_state_dict(state_dict, strict=True)

    if not resume_training:
        return {
            "iteration": 0,
            "checkpoint_iteration": int(checkpoint.get("iter", 0)),
            "best_psnr": 0.0,
            "config": checkpoint.get("config", {}),
            "checkpoint_version": checkpoint.get("checkpoint_version"),
            "model_name": checkpoint.get("model_name"),
            "metadata": checkpoint.get("metadata", {}),
            "data_state": checkpoint.get("data_state"),
        }

    if optimizer is None or "optimizer" not in checkpoint:
        raise ValueError("Checkpoint does not contain resumable optimizer state")
    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler_state = checkpoint.get("scheduler")
    restore_scheduler_state(scheduler, scheduler_state)
    scaler_state = checkpoint.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    rng_states = checkpoint.get("rng_states")
    if rng_states:
        if rng_rank >= len(rng_states):
            raise ValueError(
                f"Checkpoint has {len(rng_states)} RNG states, "
                f"but rank {rng_rank} was requested"
            )
        restore_rng_state(rng_states[rng_rank])
    else:
        restore_rng_state(checkpoint.get("rng_state"))
    return {
        "iteration": int(checkpoint.get("iter", 0)),
        "checkpoint_iteration": int(checkpoint.get("iter", 0)),
        "best_psnr": float(checkpoint.get("best_psnr", 0.0)),
        "config": checkpoint.get("config", {}),
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "model_name": checkpoint.get("model_name"),
        "metadata": checkpoint.get("metadata", {}),
        "data_state": checkpoint.get("data_state"),
    }
