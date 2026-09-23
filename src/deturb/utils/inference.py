"""Bounded-window DeTurb video inference with complete spatial/temporal coverage."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from deturb.model import DeTurb, DeTurbContract, reference_window_for_output
from .checkpoints import load_training_checkpoint
from .dynamic_inference import compute_temporal_windows, compute_tile_starts
from .numerics import require_finite


@dataclass
class InferenceStats:
    windows: int = 0
    tiles: int = 0
    output_frames: int = 0
    max_buffered_frames: int = 0
    model_seconds: float = 0.0


@dataclass
class RestoredFrame:
    index: int
    restored: torch.Tensor  # CPU float RGB CHW, before quantization
    source: torch.Tensor  # CPU RGB CHW, uint8 or float according to input_format


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("DeTurb inference supports CPU/CUDA; use --device cpu on a Mac without CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    return device


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: torch.device):
    model = DeTurb.from_config(config_path).to(device)
    state = load_training_checkpoint(
        checkpoint_path, model, torch.device("cpu"), resume_training=False,
        expected_model_name=model.contract.model_id,
        minimum_checkpoint_version=model.checkpoint_version,
        expected_model_spec=model.model_spec(),
    )
    model.eval()
    return model, state


def tiled_forward(
    clip: torch.Tensor, model: torch.nn.Module, contract: DeTurbContract,
    device: torch.device, *, patch_size: int = 128, overlap: int = 32,
    stats: InferenceStats | None = None,
) -> torch.Tensor:
    """Blend overlapping spatial tiles; return a CPU BCHW/BCTHW tensor."""
    contract.validate_model_input(clip)
    if clip.shape[0] != 1:
        raise ValueError("Video window inference expects batch size one")
    if patch_size < 32 or patch_size % 32 or not 0 <= overlap < patch_size:
        raise ValueError("patch_size must be a positive multiple of 32; 0 <= overlap < patch_size")
    require_finite((clip,), "inference input", clip.device)
    stats = stats if stats is not None else InferenceStats()
    height, width = clip.shape[-2:]
    padded_h = max(patch_size, math.ceil(height / 8) * 8)
    padded_w = max(patch_size, math.ceil(width / 8) * 8)
    padded = F.pad(clip.cpu(), (0, padded_w - width, 0, padded_h - height, 0, 0), mode="replicate")
    leading = (1, contract.output_channels, contract.input_frames) if contract.returns_full_clip else (1, contract.output_channels)
    output = torch.zeros((*leading, padded_h, padded_w), dtype=torch.float32)
    count = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for y in compute_tile_starts(padded_h, patch_size, overlap):
            for x in compute_tile_starts(padded_w, patch_size, overlap):
                tile = padded[..., y:y + patch_size, x:x + patch_size].to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                prediction = model(tile)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stats.model_seconds += time.perf_counter() - started
                stats.tiles += 1
                if not isinstance(prediction, torch.Tensor) or prediction.shape != (*leading, patch_size, patch_size):
                    raise ValueError("Model returned an output inconsistent with the DeTurb contract")
                require_finite((prediction,), "inference output", device)
                output[..., y:y + patch_size, x:x + patch_size] += prediction.float().cpu()
                count[y:y + patch_size, x:x + patch_size] += 1
    if (count == 0).any():
        raise RuntimeError("Spatial tiles left uncovered pixels")
    blended = (output / count)[..., :height, :width].contiguous()
    require_finite((blended,), "spatial tile blending", torch.device("cpu"))
    return blended


def iter_restored_frames(
    frames: Iterable[np.ndarray | torch.Tensor], total_frames: int, model: torch.nn.Module,
    contract: DeTurbContract, device: torch.device, *, patch_size: int = 128,
    overlap: int = 32, stats: InferenceStats | None = None,
    input_format: str = "bgr_uint8",
) -> Iterator[RestoredFrame]:
    """Consume BGR uint8 or RGB float frames; emit each index once, in order.

    Clip mode uses overlapping clips and disjoint center output intervals.
    Reference mode uses a replicate-padded window centered on each output.
    The source-frame cache contains at most input_frames images.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if input_format not in ("bgr_uint8", "rgb_float"):
        raise ValueError("input_format must be bgr_uint8 or rgb_float")
    stats = stats if stats is not None else InferenceStats()
    def plans():
        if contract.returns_full_clip:
            for window in compute_temporal_windows(total_frames, contract.input_frames):
                indices = tuple(min(window.input_start + t, total_frames - 1) for t in range(contract.input_frames))
                yield indices, tuple((i, i - window.input_start) for i in range(window.output_start, window.output_end))
        else:
            for index in range(total_frames):
                window = reference_window_for_output(total_frames, index, contract)
                yield window.input_indices, ((index, contract.reference_index),)
    iterator = iter(frames)
    cache: dict[int, torch.Tensor] = {}
    next_read, next_output = 0, 0
    shape = None
    for indices, outputs in plans():
        first_needed = min(indices)
        cache = {i: frame for i, frame in cache.items() if i >= first_needed}
        while next_read <= max(indices):
            try:
                bgr = next(iterator)
            except StopIteration as error:
                raise OSError(f"Video ended before required frame {next_read}") from error
            if input_format == "bgr_uint8":
                if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
                    raise ValueError("Video frames must be BGR uint8 HWC images")
                rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)
            else:
                if not isinstance(bgr, torch.Tensor) or bgr.ndim != 3 or bgr.shape[0] != 3 or not bgr.is_floating_point():
                    raise ValueError("Float frames must be RGB CHW tensors")
                rgb = bgr.detach().cpu().float()
                require_finite((rgb,), "float video input", torch.device("cpu"))
            if shape is None:
                shape = bgr.shape
            if bgr.shape != shape:
                raise ValueError("Video frame dimensions changed")
            if next_read >= first_needed:
                cache[next_read] = rgb
            next_read += 1
        stats.max_buffered_frames = max(stats.max_buffered_frames, len(cache))
        clip = torch.stack([cache[i] for i in indices], dim=1).unsqueeze(0).float()
        if input_format == "bgr_uint8":
            clip.div_(255)
        prediction = tiled_forward(clip, model, contract, device, patch_size=patch_size, overlap=overlap, stats=stats)
        stats.windows += 1
        for index, position in outputs:
            if index != next_output:
                raise RuntimeError("Temporal windows duplicated or skipped an output frame")
            image = prediction[0, :, position] if contract.returns_full_clip else prediction[0]
            next_output += 1
            stats.output_frames += 1
            yield RestoredFrame(index, image.clone(), cache[index])
    if next_output != total_frames:
        raise RuntimeError("Temporal inference did not restore the complete video")


def decoded_frames(path: str | Path, count: int, start: int = 0) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {path}")
        if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise OSError(f"Cannot seek to frame {start}: {path}")
        for index in range(start, start + count):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OSError(f"Cannot decode frame {index}: {path}")
            yield frame
    finally:
        capture.release()
