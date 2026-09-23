"""Runtime and device helpers for DeTurb."""

from __future__ import annotations

import torch


def resolve_device(device_name: str = "auto") -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_amp_dtype(
    amp_name: str,
    device: torch.device,
) -> torch.dtype | None:
    normalized = amp_name.lower()
    if normalized == "none":
        return None
    if normalized == "bf16":
        if device.type not in ("cuda", "cpu"):
            raise ValueError("BF16 autocast is supported here only on CUDA or CPU")
        return torch.bfloat16
    if normalized == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 training autocast requires CUDA")
        return torch.float16
    raise ValueError("amp dtype must be one of: none, bf16, fp16")
