"""Pure-PyTorch deformable 3D convolution for the DeTurb registration net.

The implementation intentionally favors an explicit, testable mathematical
contract before a custom CUDA optimization. Inputs use ``[B, C, T, H, W]``.
Offsets use grouped ``(dt, dy, dx)`` channel blocks with shape
``[B, deform_groups * 3 * K, out_T, out_H, out_W]`` where
``K = kT * kH * kW``.

Offsets follow backward-sampling semantics: positive values sample farther
along the corresponding input axis and therefore move visible content in the
negative output direction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F


Triple = tuple[int, int, int]


def _triple(value: int | Sequence[int], name: str, *, allow_zero: bool) -> Triple:
    if isinstance(value, Integral) and not isinstance(value, bool):
        scalar = int(value)
        result = (scalar, scalar, scalar)
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
        and all(isinstance(item, Integral) and not isinstance(item, bool) for item in value)
    ):
        result = tuple(int(item) for item in value)
    else:
        raise TypeError(f"{name} must be an int or a length-3 sequence")
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in result):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {qualifier}")
    return result  # type: ignore[return-value]


def _output_shape(
    input_shape: Triple,
    kernel_size: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> Triple:
    output = tuple(
        (input_size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for input_size, kernel, step, pad, dil in zip(
            input_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        )
    )
    if any(size <= 0 for size in output):
        raise ValueError(
            "kernel/stride/padding/dilation produce a non-positive output shape"
        )
    return output  # type: ignore[return-value]


def _normalize_coordinate(
    coordinate: torch.Tensor,
    size: int,
    align_corners: bool,
) -> torch.Tensor:
    if size == 1:
        return coordinate.mul(2.0)
    if align_corners:
        return coordinate.mul(2.0 / (size - 1)).sub(1.0)
    return coordinate.add(0.5).mul(2.0 / size).sub(1.0)


def _validate_deform_conv3d(
    input_tensor: torch.Tensor,
    offsets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
    groups: int,
    deform_groups: int,
) -> tuple[int, int, int, int, int, Triple, Triple]:
    for name, tensor, dimensions in (
        ("input", input_tensor, 5),
        ("offsets", offsets, 5),
        ("weight", weight, 5),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != dimensions:
            raise ValueError(f"{name} must be {dimensions}D")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.device != input_tensor.device or tensor.dtype != input_tensor.dtype:
            raise ValueError("input, offsets and weight must share device and dtype")

    batch, input_channels, input_t, input_h, input_w = input_tensor.shape
    output_channels, weight_channels, kernel_t, kernel_h, kernel_w = weight.shape
    if min(batch, input_channels, input_t, input_h, input_w, output_channels) <= 0:
        raise ValueError("input and weight dimensions must be positive")
    if groups <= 0 or deform_groups <= 0:
        raise ValueError("groups and deform_groups must be positive")
    if input_channels % groups or output_channels % groups:
        raise ValueError("input and output channels must be divisible by groups")
    if input_channels % deform_groups:
        raise ValueError("input channels must be divisible by deform_groups")
    if weight_channels != input_channels // groups:
        raise ValueError(
            "weight input channels must equal input_channels // groups"
        )

    kernel_size = (kernel_t, kernel_h, kernel_w)
    output_shape = _output_shape(
        (input_t, input_h, input_w),
        kernel_size,
        stride,
        padding,
        dilation,
    )
    kernel_points = kernel_t * kernel_h * kernel_w
    expected_offsets = (
        batch,
        deform_groups * 3 * kernel_points,
        *output_shape,
    )
    if tuple(offsets.shape) != expected_offsets:
        raise ValueError(
            f"offsets must have shape {expected_offsets}, got {tuple(offsets.shape)}"
        )
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
            raise ValueError("bias must be a 1D torch.Tensor")
        if bias.shape[0] != output_channels:
            raise ValueError("bias length must equal output channels")
        if bias.device != input_tensor.device or bias.dtype != input_tensor.dtype:
            raise ValueError("bias must share input device and dtype")
    return (
        batch,
        input_channels,
        output_channels,
        kernel_points,
        input_channels // deform_groups,
        output_shape,
        kernel_size,
    )


def deform_conv3d(
    input_tensor: torch.Tensor,
    offsets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    deform_groups: int = 1,
    sampling_chunk_size: int | None = 16,
    align_corners: bool = False,
) -> torch.Tensor:
    """Apply deformable 3D convolution using differentiable grid sampling.

    ``sampling_chunk_size=None`` samples all kernel points together and is the
    simple reference path. A positive chunk size bounds intermediate memory and
    is intended for training.
    """

    stride_3 = _triple(stride, "stride", allow_zero=False)
    padding_3 = _triple(padding, "padding", allow_zero=True)
    dilation_3 = _triple(dilation, "dilation", allow_zero=False)
    if sampling_chunk_size is not None and sampling_chunk_size <= 0:
        raise ValueError("sampling_chunk_size must be positive or None")

    (
        batch,
        input_channels,
        output_channels,
        kernel_points,
        channels_per_deform_group,
        output_shape,
        kernel_size,
    ) = _validate_deform_conv3d(
        input_tensor,
        offsets,
        weight,
        bias,
        stride_3,
        padding_3,
        dilation_3,
        groups,
        deform_groups,
    )
    input_t, input_h, input_w = input_tensor.shape[-3:]
    # align_corners=True collapses every normalized coordinate on a size-one
    # axis to its sole pixel, including out-of-bounds kernel taps. The False
    # convention represents pixel coordinates correctly on all axes instead.
    sampling_align_corners = align_corners and min(input_t, input_h, input_w) > 1
    output_t, output_h, output_w = output_shape
    chunk_size = kernel_points if sampling_chunk_size is None else min(
        sampling_chunk_size,
        kernel_points,
    )

    kernel_t, kernel_h, kernel_w = kernel_size
    kernel_grid = torch.meshgrid(
        torch.arange(kernel_t, device=input_tensor.device, dtype=input_tensor.dtype),
        torch.arange(kernel_h, device=input_tensor.device, dtype=input_tensor.dtype),
        torch.arange(kernel_w, device=input_tensor.device, dtype=input_tensor.dtype),
        indexing="ij",
    )
    kernel_coordinates = tuple(
        coordinate.reshape(-1) * dil
        for coordinate, dil in zip(kernel_grid, dilation_3)
    )
    output_grid = torch.meshgrid(
        torch.arange(output_t, device=input_tensor.device, dtype=input_tensor.dtype),
        torch.arange(output_h, device=input_tensor.device, dtype=input_tensor.dtype),
        torch.arange(output_w, device=input_tensor.device, dtype=input_tensor.dtype),
        indexing="ij",
    )
    output_origins = tuple(
        coordinate * step - pad
        for coordinate, step, pad in zip(output_grid, stride_3, padding_3)
    )

    offset_view = offsets.reshape(
        batch,
        deform_groups,
        3,
        kernel_points,
        output_t,
        output_h,
        output_w,
    )
    flat_weight = weight.reshape(
        output_channels,
        input_channels // groups,
        kernel_points,
    )
    input_channels_per_group = input_channels // groups
    output_channels_per_group = output_channels // groups
    output: torch.Tensor | None = None

    for point_start in range(0, kernel_points, chunk_size):
        point_end = min(point_start + chunk_size, kernel_points)
        points = point_end - point_start
        sampled_deform_groups = []

        for deform_group in range(deform_groups):
            channel_start = deform_group * channels_per_deform_group
            channel_end = channel_start + channels_per_deform_group
            group_input = input_tensor[:, channel_start:channel_end]
            group_offsets = offset_view[
                :,
                deform_group,
                :,
                point_start:point_end,
            ]

            sample_t = (
                output_origins[0].unsqueeze(0).unsqueeze(0)
                + kernel_coordinates[0][point_start:point_end].view(1, points, 1, 1, 1)
                + group_offsets[:, 0]
            )
            sample_y = (
                output_origins[1].unsqueeze(0).unsqueeze(0)
                + kernel_coordinates[1][point_start:point_end].view(1, points, 1, 1, 1)
                + group_offsets[:, 1]
            )
            sample_x = (
                output_origins[2].unsqueeze(0).unsqueeze(0)
                + kernel_coordinates[2][point_start:point_end].view(1, points, 1, 1, 1)
                + group_offsets[:, 2]
            )
            sampling_grid = torch.stack(
                (
                    _normalize_coordinate(sample_x, input_w, sampling_align_corners),
                    _normalize_coordinate(sample_y, input_h, sampling_align_corners),
                    _normalize_coordinate(sample_t, input_t, sampling_align_corners),
                ),
                dim=-1,
            )

            expanded_input = group_input.unsqueeze(1).expand(
                batch,
                points,
                channels_per_deform_group,
                input_t,
                input_h,
                input_w,
            ).reshape(
                batch * points,
                channels_per_deform_group,
                input_t,
                input_h,
                input_w,
            )
            sampled = F.grid_sample(
                expanded_input,
                sampling_grid.reshape(
                    batch * points,
                    output_t,
                    output_h,
                    output_w,
                    3,
                ),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=sampling_align_corners,
            ).reshape(
                batch,
                points,
                channels_per_deform_group,
                output_t,
                output_h,
                output_w,
            ).permute(0, 2, 1, 3, 4, 5)
            sampled_deform_groups.append(sampled)

        sampled_input = torch.cat(sampled_deform_groups, dim=1)
        chunk_outputs = []
        for group in range(groups):
            input_start = group * input_channels_per_group
            input_end = input_start + input_channels_per_group
            output_start = group * output_channels_per_group
            output_end = output_start + output_channels_per_group
            chunk_outputs.append(
                torch.einsum(
                    "bckthw,ock->bothw",
                    sampled_input[:, input_start:input_end],
                    flat_weight[
                        output_start:output_end,
                        :,
                        point_start:point_end,
                    ],
                )
            )
        chunk_output = torch.cat(chunk_outputs, dim=1)
        output = chunk_output if output is None else output + chunk_output

    if output is None:
        raise RuntimeError("deformable convolution sampled no kernel points")
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1, 1)
    return output


def deform_conv3d_reference(
    input_tensor: torch.Tensor,
    offsets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Full-kernel reference path used for correctness and gradient tests."""

    if "sampling_chunk_size" in kwargs:
        raise TypeError("reference path controls sampling_chunk_size")
    return deform_conv3d(
        input_tensor,
        offsets,
        weight,
        bias,
        sampling_chunk_size=None,
        **kwargs,
    )


class DeformConv3d(nn.Module):
    """Learn offsets and apply a memory-bounded deformable 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | None = None,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
        sampling_chunk_size: int | None = 16,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size, "kernel_size", allow_zero=False)
        self.stride = _triple(stride, "stride", allow_zero=False)
        self.dilation = _triple(dilation, "dilation", allow_zero=False)
        if padding is None:
            self.padding = tuple(
                dil * (kernel - 1) // 2
                for kernel, dil in zip(self.kernel_size, self.dilation)
            )
        else:
            self.padding = _triple(padding, "padding", allow_zero=True)
        if groups <= 0 or deform_groups <= 0:
            raise ValueError("groups and deform_groups must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("channels must be divisible by groups")
        if in_channels % deform_groups:
            raise ValueError("in_channels must be divisible by deform_groups")
        if sampling_chunk_size is not None and sampling_chunk_size <= 0:
            raise ValueError("sampling_chunk_size must be positive or None")
        self.groups = groups
        self.deform_groups = deform_groups
        self.sampling_chunk_size = sampling_chunk_size
        self.align_corners = align_corners

        kernel_points = math.prod(self.kernel_size)
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                *self.kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.offset_conv = nn.Conv3d(
            in_channels,
            deform_groups * 3 * kernel_points,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (self.in_channels // self.groups) * math.prod(self.kernel_size)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def predict_offsets(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.offset_conv(input_tensor)

    def forward(
        self,
        input_tensor: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if offsets is None:
            offsets = self.predict_offsets(input_tensor)
        return deform_conv3d(
            input_tensor,
            offsets,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            deform_groups=self.deform_groups,
            sampling_chunk_size=self.sampling_chunk_size,
            align_corners=self.align_corners,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, deform_groups={self.deform_groups}, "
            f"sampling_chunk_size={self.sampling_chunk_size}"
        )


class DeformConv3dReference(DeformConv3d):
    """Module form of the unchunked correctness reference implementation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        if "sampling_chunk_size" in kwargs:
            raise TypeError("DeformConv3dReference controls sampling_chunk_size")
        super().__init__(*args, sampling_chunk_size=None, **kwargs)
