"""Optional DistributedDataParallel runtime helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from .runtime import resolve_device


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def broadcast_object(self, value: Any, source: int = 0) -> Any:
        if not self.distributed:
            return value
        objects = [value if self.rank == source else None]
        kwargs = {"device": self.device} if self.device.type == "cuda" else {}
        dist.broadcast_object_list(objects, src=source, **kwargs)
        return objects[0]

    def reduce_sums(self, values: Sequence[float | int]) -> list[float]:
        tensor_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        tensor = torch.tensor(values, dtype=torch.float64, device=tensor_device)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.distributed:
            return [value]
        objects: list[Any] = [None] * self.world_size
        dist.all_gather_object(objects, value)
        return objects

    def close(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(
    device_name: str = "auto", *, timeout_seconds: int = 120,
) -> DistributedContext:
    if timeout_seconds <= 0:
        raise ValueError("distributed timeout must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=resolve_device(device_name),
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    normalized_device = device_name.lower()
    if torch.cuda.is_available():
        if normalized_device not in ("auto", "cuda"):
            raise ValueError(
                "GPU DDP requires --device auto or --device cuda; "
                "LOCAL_RANK selects the concrete GPU"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = resolve_device(device_name)
        if device.type != "cpu":
            raise RuntimeError("Non-CUDA DDP tests require --device cpu")
        backend = "gloo"

    init_kwargs = {
        "backend": backend,
        "init_method": "env://",
        "rank": rank,
        "world_size": world_size,
        "timeout": timedelta(seconds=timeout_seconds),
    }
    if device.type == "cuda":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    return DistributedContext(
        distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without padding or duplicating samples."""

    def __init__(
        self,
        dataset: Dataset[Any],
        rank: int,
        world_size: int,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = max(len(self.dataset) - self.rank, 0)
        return math.ceil(remaining / self.world_size)


def close_distributed_if_initialized() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
