"""Reconstructible batch order and sample augmentation, independent of prefetch."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset


DATA_POLICY = "indexed_sample_v1"


def paired_dataset_fingerprint(dataset: Dataset) -> str:
    """Fingerprint ordered pairs and file metadata without reading video contents."""
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [int(dataset.indices[i]) for i in indices]
        dataset = dataset.dataset
    digest = hashlib.sha256()
    fields = ('gt_list', 'turb_list') if getattr(dataset, 'data_layout', 'triplet') == 'paired_turb' else ('gt_list', 'turb_list', 'blur_list')
    for index in indices:
        paths = (dataset.fingerprint_paths(index) if hasattr(dataset, 'fingerprint_paths')
                 else [getattr(dataset, field)[index] for field in fields])
        for item in paths:
            path = Path(item).expanduser().resolve()
            stat = path.stat()
            digest.update(json.dumps([str(path), stat.st_size, stat.st_mtime_ns]).encode())
            digest.update(b"\n")
    return digest.hexdigest()


class IndexedAugmentationDataset(Dataset):
    """Run one sample with an isolated Python/NumPy/CPU-Torch RNG state.

    Requests contain (sample_index, augmentation_seed), so persistent workers
    never need to observe a mutable epoch variable or serialize their queues.
    CUDA RNG is deliberately untouched when num_workers=0.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, request):
        index, seed = request
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.set_rng_state(generator.get_state())
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


class IndexedBatchSampler(Sampler):
    """Epoch-shuffled rank shards with a cursor counting completed optimizer steps."""

    def __init__(self, dataset_size: int, batch_size: int, *, seed: int,
                 rank: int = 0, world_size: int = 1):
        if min(dataset_size, batch_size, world_size) <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid indexed sampler dimensions/rank")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.samples_per_rank = dataset_size // world_size
        self.batches_per_epoch = self.samples_per_rank // batch_size
        if self.batches_per_epoch == 0:
            raise ValueError("indexed sampler has no complete batch per rank")
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int, start_batch: int = 0):
        if epoch < 0 or not 0 <= start_batch <= self.batches_per_epoch:
            raise ValueError("invalid indexed sampler epoch/batch cursor")
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self):
        return self.batches_per_epoch - self.start_batch

    def __iter__(self):
        epoch, start_batch = self.epoch, self.start_batch
        generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
        indices = torch.randperm(self.dataset_size, generator=generator, device="cpu").tolist()
        indices = indices[:self.samples_per_rank * self.world_size][self.rank::self.world_size]
        for batch in range(start_batch, self.batches_per_epoch):
            requests = []
            for position in range(batch * self.batch_size, (batch + 1) * self.batch_size):
                index = indices[position]
                key = f"{self.seed}:{epoch}:{self.rank}:{position}:{index}".encode()
                seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & ((1 << 63) - 1)
                requests.append((index, seed))
            yield requests

    def spec(self, fingerprint: str) -> dict:
        return {"policy": DATA_POLICY, "dataset_size": self.dataset_size,
                "batch_size": self.batch_size, "world_size": self.world_size, "seed": self.seed,
                "batches_per_epoch": self.batches_per_epoch, "dataset_fingerprint": fingerprint}

    def state_dict(self, next_iteration: int, fingerprint: str) -> dict:
        return {**self.spec(fingerprint), "next_iteration": next_iteration,
                "epoch": next_iteration // self.batches_per_epoch,
                "batch_in_epoch": next_iteration % self.batches_per_epoch}


def validate_data_state(state: dict | None, iteration: int, expected_spec: dict | None = None):
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("Indexed data iteration must be a non-negative integer")
    if not isinstance(state, dict) or state.get("policy") != DATA_POLICY:
        raise ValueError("Checkpoint lacks indexed data state; use --finetune or explicit --data-order legacy")
    fields = ("dataset_size", "batch_size", "world_size", "batches_per_epoch", "next_iteration", "epoch", "batch_in_epoch", "seed")
    if any(not isinstance(state.get(key), int) or isinstance(state.get(key), bool) for key in fields):
        raise ValueError("Checkpoint indexed data state is incomplete")
    if min(state["dataset_size"], state["batch_size"], state["world_size"], state["batches_per_epoch"]) <= 0:
        raise ValueError("Checkpoint indexed data dimensions are invalid")
    batches = state["dataset_size"] // state["world_size"] // state["batch_size"]
    if state["batches_per_epoch"] != batches or state["next_iteration"] != iteration or (
        state["epoch"], state["batch_in_epoch"]
    ) != divmod(iteration, batches):
        raise ValueError("Checkpoint indexed data cursor disagrees with completed iterations")
    if not isinstance(state.get("dataset_fingerprint"), str) or not state["dataset_fingerprint"]:
        raise ValueError("Checkpoint dataset fingerprint is missing")
    if expected_spec is not None and any(state.get(key) != value for key, value in expected_spec.items()):
        raise ValueError("Checkpoint data order/dataset differs; use --finetune for a new data stream")
