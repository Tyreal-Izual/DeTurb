"""Temporal window and dataset-layout utilities for DeTurb."""

from __future__ import annotations

import torch

from .config import DeTurbContract
from .types import TemporalReferenceWindow


def dataset_clip_to_model_input(
    sequence: torch.Tensor,
    contract: DeTurbContract,
) -> torch.Tensor:
    """Convert a DataLoader batch from ``[B, T, C, H, W]`` to BCTHW."""

    _validate_dataset_sequence(sequence, contract, "sequence")
    return sequence.permute(0, 2, 1, 3, 4).contiguous()


def select_reference_target(
    target_sequence: torch.Tensor,
    contract: DeTurbContract,
) -> torch.Tensor:
    """Select the configured clean target from a ``[B, T, C, H, W]`` batch."""

    _validate_dataset_sequence(target_sequence, contract, "target_sequence")
    return target_sequence[:, contract.reference_index, :, :, :].contiguous()


def select_output_target(
    target_sequence: torch.Tensor,
    contract: DeTurbContract,
) -> torch.Tensor:
    """Keep every time-matched GT frame for clip training, or select the legacy reference."""
    if contract.returns_full_clip:
        return dataset_clip_to_model_input(target_sequence, contract)
    return select_reference_target(target_sequence, contract)


def _validate_dataset_sequence(
    sequence: torch.Tensor,
    contract: DeTurbContract,
    name: str,
) -> None:
    if not isinstance(sequence, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if sequence.ndim != 5:
        raise ValueError(f"{name} must have shape [B, T, C, H, W]")
    batch, frames, channels, height, width = sequence.shape
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    if frames != contract.input_frames:
        raise ValueError(
            f"{name} has {frames} frames; expected {contract.input_frames}"
        )
    if channels != contract.input_channels:
        raise ValueError(
            f"{name} has {channels} channels; expected {contract.input_channels}"
        )
    if not torch.is_floating_point(sequence):
        raise TypeError(f"{name} must use a floating-point dtype")


def reference_window_for_output(
    total_frames: int,
    output_index: int,
    contract: DeTurbContract,
) -> TemporalReferenceWindow:
    """Build a replicate-padded input window for one output frame."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not 0 <= output_index < total_frames:
        raise ValueError("output_index is outside the video")
    if contract.temporal_padding != "replicate":
        raise ValueError(
            f"unsupported temporal padding: {contract.temporal_padding}"
        )

    first_index = output_index - contract.reference_index
    indices = tuple(
        min(max(first_index + position, 0), total_frames - 1)
        for position in range(contract.input_frames)
    )
    return TemporalReferenceWindow(
        output_index=output_index,
        input_indices=indices,
        reference_index=contract.reference_index,
    )


def build_reference_windows(
    total_frames: int,
    contract: DeTurbContract,
) -> tuple[TemporalReferenceWindow, ...]:
    """Return exactly one model window for every output video frame."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    return tuple(
        reference_window_for_output(total_frames, output_index, contract)
        for output_index in range(total_frames)
    )


def materialize_model_window(
    frames: torch.Tensor,
    window: TemporalReferenceWindow,
) -> torch.Tensor:
    """Gather ``[F, C, H, W]`` frames into one ``[C, T, H, W]`` clip."""

    if not isinstance(frames, torch.Tensor):
        raise TypeError("frames must be a torch.Tensor")
    if frames.ndim != 4:
        raise ValueError("frames must have shape [F, C, H, W]")
    if frames.shape[0] <= 0:
        raise ValueError("frames cannot be empty")
    if min(window.input_indices) < 0 or max(window.input_indices) >= frames.shape[0]:
        raise ValueError("window contains a source index outside frames")
    indices = torch.tensor(
        window.input_indices,
        dtype=torch.long,
        device=frames.device,
    )
    return frames.index_select(0, indices).permute(1, 0, 2, 3).contiguous()
