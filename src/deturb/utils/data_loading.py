"""Deterministic DataLoader worker and prefetch configuration."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def worker_options(
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if num_workers == 0:
        return {}
    return {
        "worker_init_fn": seed_worker,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }
