"""Differentiable frame-wise 2D flow operations for DeTurb."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


PaddingMode = Literal["zeros", "border", "reflection"]
InterpolationMode = Literal["bilinear", "nearest", "bicubic"]


def _validate_frames_and_flow(
    frames: torch.Tensor,
    flow: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if not isinstance(frames, torch.Tensor) or not isinstance(flow, torch.Tensor):
        raise TypeError("frames and flow must be torch.Tensor instances")
    if frames.ndim != 5:
        raise ValueError("frames must have shape [B, C, T, H, W]")
    if flow.ndim != 5:
        raise ValueError("flow must have shape [B, T, 2, H, W]")
    batch, channels, frames_count, height, width = frames.shape
    if min(batch, channels, frames_count, height, width) <= 0:
        raise ValueError("frames dimensions must be positive")
    if flow.shape != (batch, frames_count, 2, height, width):
        raise ValueError(
            "flow must have shape "
            f"{(batch, frames_count, 2, height, width)}, got {tuple(flow.shape)}"
        )
    if not torch.is_floating_point(frames) or not torch.is_floating_point(flow):
        raise TypeError("frames and flow must use floating-point dtypes")
    if frames.device != flow.device:
        raise ValueError("frames and flow must be on the same device")
    if frames.dtype != flow.dtype:
        raise ValueError("frames and flow must use the same dtype")
    return batch, channels, frames_count, height, width


def _normalize_coordinate(
    coordinate: torch.Tensor,
    size: int,
    align_corners: bool,
) -> torch.Tensor:
    if size == 1:
        # Keep the sole valid pixel at zero while still mapping non-zero pixel
        # offsets outside the image. Returning all zeros would incorrectly
        # turn every displacement into an in-bounds sample.
        return coordinate.mul(2.0)
    if align_corners:
        return coordinate.mul(2.0 / (size - 1)).sub(1.0)
    return coordinate.add(0.5).mul(2.0 / size).sub(1.0)


def warp_frames_2d(
    frames: torch.Tensor,
    flow: torch.Tensor,
    *,
    mode: InterpolationMode = "bilinear",
    padding_mode: PaddingMode = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """Warp every video frame with a pixel-unit backward sampling flow.

    ``frames`` uses ``[B, C, T, H, W]`` and ``flow`` uses
    ``[B, T, 2, H, W]``. Flow channels are ``(dx, dy)``. The definition is

    ``output[..., y, x] = input[..., y + dy, x + dx]``.

    Consequently, a positive constant ``dx`` samples from the right and moves
    visible content to the left. The operation is differentiable with respect
    to both frames and flow.
    """

    batch, channels, frames_count, height, width = _validate_frames_and_flow(
        frames,
        flow,
    )
    if mode not in {"bilinear", "nearest", "bicubic"}:
        raise ValueError(f"unsupported interpolation mode: {mode}")
    if padding_mode not in {"zeros", "border", "reflection"}:
        raise ValueError(f"unsupported padding mode: {padding_mode}")

    frame_batch = frames.permute(0, 2, 1, 3, 4).reshape(
        batch * frames_count,
        channels,
        height,
        width,
    )
    flow_batch = flow.reshape(batch * frames_count, 2, height, width)

    y = torch.arange(height, device=frames.device, dtype=frames.dtype)
    x = torch.arange(width, device=frames.device, dtype=frames.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    sample_x = grid_x.unsqueeze(0) + flow_batch[:, 0]
    sample_y = grid_y.unsqueeze(0) + flow_batch[:, 1]
    sampling_height, sampling_width = height, width
    if align_corners and min(height, width) == 1:
        # Give singleton axes real coordinate extent. Zero padding retains
        # out-of-bounds taps; repeated pixels retain border/reflection behavior.
        # Keeping align_corners=True also preserves reflection of bicubic taps
        # on the other, non-singleton axis. An even origin shift preserves
        # nearest interpolation's round-to-even choice at half-pixel offsets.
        pad_x, pad_y = 2 * int(width == 1), 2 * int(height == 1)
        frame_batch = F.pad(frame_batch, (pad_x, pad_x, pad_y, pad_y),
                            mode="constant" if padding_mode == "zeros" else "replicate")
        sample_x = sample_x + pad_x
        sample_y = sample_y + pad_y
        sampling_width += 2 * pad_x
        sampling_height += 2 * pad_y
    grid = torch.stack(
        (
            _normalize_coordinate(sample_x, sampling_width, align_corners),
            _normalize_coordinate(sample_y, sampling_height, align_corners),
        ),
        dim=-1,
    )

    warped = F.grid_sample(
        frame_batch,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return warped.reshape(
        batch,
        frames_count,
        channels,
        height,
        width,
    ).permute(0, 2, 1, 3, 4).contiguous()


def _validate_flow(flow: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(flow, torch.Tensor):
        raise TypeError("flow must be a torch.Tensor")
    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("flow must have shape [B, T, 2, H, W]")
    batch, frames_count, _, height, width = flow.shape
    if min(batch, frames_count, height, width) <= 0:
        raise ValueError("flow dimensions must be positive")
    if not torch.is_floating_point(flow):
        raise TypeError("flow must use a floating-point dtype")
    return batch, frames_count, height, width


def resize_flow_2d(
    flow: torch.Tensor,
    size: tuple[int, int],
    *,
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize a pixel-unit flow and scale displacement magnitudes."""

    batch, frames_count, source_height, source_width = _validate_flow(flow)
    target_height, target_width = size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target flow size must be positive")
    if (target_height, target_width) == (source_height, source_width):
        return flow

    flat_flow = flow.reshape(batch * frames_count, 2, source_height, source_width)
    resized = F.interpolate(
        flat_flow,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=align_corners,
    )
    if align_corners:
        scale_x = (
            (target_width - 1) / (source_width - 1)
            if source_width > 1
            else 1.0
        )
        scale_y = (
            (target_height - 1) / (source_height - 1)
            if source_height > 1
            else 1.0
        )
    else:
        scale_x = target_width / source_width
        scale_y = target_height / source_height
    scale = flow.new_tensor((scale_x, scale_y)).view(1, 2, 1, 1)
    return resized.mul(scale).reshape(
        batch,
        frames_count,
        2,
        target_height,
        target_width,
    )


def compose_backward_flows(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    padding_mode: PaddingMode = "border",
    align_corners: bool = False,
) -> torch.Tensor:
    """Compose two backward flows in sequential-warp order.

    The result corresponds geometrically to applying ``first`` to the image and
    then applying ``second``. It is
    ``second + sample(first, identity + second)``.
    """

    first_shape = _validate_flow(first)
    second_shape = _validate_flow(second)
    if first_shape != second_shape:
        raise ValueError("first and second flow shapes must match")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError("first and second flows must share device and dtype")

    first_as_frames = first.permute(0, 2, 1, 3, 4).contiguous()
    sampled_first = warp_frames_2d(
        first_as_frames,
        second,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).permute(0, 2, 1, 3, 4).contiguous()
    return second + sampled_first
