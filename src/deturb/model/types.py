"""Shared immutable types for the DeTurb model contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TemporalReferenceWindow:
    """One model window that produces exactly one output video frame.

    ``input_indices`` contains already padded source-frame indices. The frame at
    ``reference_index`` always maps to ``output_index``.
    """

    output_index: int
    input_indices: tuple[int, ...]
    reference_index: int

    def __post_init__(self) -> None:
        if self.output_index < 0:
            raise ValueError("output_index must be non-negative")
        if not self.input_indices:
            raise ValueError("input_indices cannot be empty")
        if any(index < 0 for index in self.input_indices):
            raise ValueError("input_indices must be non-negative")
        if any(
            right < left or right - left > 1
            for left, right in zip(self.input_indices, self.input_indices[1:])
        ):
            raise ValueError(
                "input_indices must be a replicate-padded consecutive window"
            )
        if not 0 <= self.reference_index < len(self.input_indices):
            raise ValueError("reference_index is outside the temporal window")
        if self.input_indices[self.reference_index] != self.output_index:
            raise ValueError(
                "the reference position must point to the output frame"
            )

    @property
    def input_frames(self) -> int:
        return len(self.input_indices)

    @property
    def padded_positions(self) -> int:
        return len(self.input_indices) - len(set(self.input_indices))


@dataclass(frozen=True, slots=True)
class RegistrationOutput:
    """Structured coarse-to-fine outputs from the DeTurb registration network."""

    aligned: torch.Tensor
    aligned_coarse: torch.Tensor
    aligned_mid: torch.Tensor
    residual_flow_coarse: torch.Tensor
    residual_flow_mid: torch.Tensor
    residual_flow_full: torch.Tensor
    cumulative_flow_coarse: torch.Tensor
    cumulative_flow_mid: torch.Tensor
    cumulative_flow_full: torch.Tensor
    # Raw final warp, before the optional learned RGB refinement.
    warped_full: torch.Tensor | None = None

    @property
    def residual_flows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.residual_flow_coarse,
            self.residual_flow_mid,
            self.residual_flow_full,
        )

    @property
    def cumulative_flows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.cumulative_flow_coarse,
            self.cumulative_flow_mid,
            self.cumulative_flow_full,
        )

    @property
    def aligned_stages(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.aligned_coarse, self.aligned_mid, self.aligned


@dataclass(frozen=True, slots=True)
class FusionOutput:
    """Contract-selected result (BCHW or BCTHW) plus the raw RGB clip."""

    restored: torch.Tensor
    predicted_clip: torch.Tensor
    padded_spatial_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class DeTurbOutput:
    """Complete auxiliary output from the final DeTurb model."""

    restored: torch.Tensor
    registration: RegistrationOutput
    fusion: FusionOutput
