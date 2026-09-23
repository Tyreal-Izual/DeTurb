"""Finite-value checks with coordinated failure for synchronous DDP steps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import cmath
import math
from numbers import Complex, Integral, Real
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def tensors_in(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensors_in(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors_in(item)


def all_finite(tensors: Iterable[torch.Tensor]) -> bool:
    checks: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            checks.setdefault(tensor.device, []).append(torch.isfinite(tensor.detach()).all())
    return all(bool(torch.stack(values).all().item()) for values in checks.values())


def _scalar_state_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True  # Checked together by device, without a sync per tensor.
    if isinstance(value, Mapping):
        return all(_scalar_state_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_scalar_state_finite(v) for v in value)
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return bool(np.isfinite(value).all())
    if isinstance(value, Integral):
        return True
    if isinstance(value, Real):
        return math.isfinite(value)
    if isinstance(value, Complex):
        return cmath.isfinite(value)
    return True


def state_is_finite(value: Any) -> bool:
    """Include scalar learning rates, scheduler counters and NumPy state arrays."""
    return _scalar_state_finite(value) and all_finite(tensors_in(value))


def raise_if_any_rank_failed(
    failed: bool, stage: str, device: torch.device, *, synchronize: bool = True,
) -> None:
    if synchronize and dist.is_available() and dist.is_initialized():
        flag_device = device if dist.get_backend() == "nccl" else torch.device("cpu")
        flag = torch.tensor(int(failed), dtype=torch.int32, device=flag_device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        failed = bool(flag.item())
    if failed:
        raise FloatingPointError(f"Non-finite values detected during {stage}; training/evaluation stopped")


def require_finite(
    tensors: Iterable[torch.Tensor], stage: str, device: torch.device,
    *, synchronize: bool = False,
) -> None:
    raise_if_any_rank_failed(not all_finite(tensors), stage, device, synchronize=synchronize)


def require_finite_state(value: Any, stage: str, device: torch.device, *, synchronize: bool = False) -> None:
    raise_if_any_rank_failed(not state_is_finite(value), stage, device, synchronize=synchronize)
