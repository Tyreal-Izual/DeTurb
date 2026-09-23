"""Paper-aligned multi-scale non-rigid registration for DeTurb."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeTurbContract, RegistrationConfig
from .deform_conv3d import DeformConv3d
from .types import RegistrationOutput
from .warping import compose_backward_flows, resize_flow_2d, warp_frames_2d


def _same_padding(kernel: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(size // 2 for size in kernel)  # type: ignore[return-value]


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported registration activation: {name}")


class ConvActivation3d(nn.Module):
    """One regular 3D convolution followed by the paper's ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        activation: str,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=_same_padding(kernel_size),
        )
        self.activation = _activation(activation)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(input_tensor))


class DeformActivation3d(nn.Module):
    """One deformable 3D convolution followed by ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        *,
        deform_groups: int,
        sampling_chunk_size: int,
        align_corners: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.conv = DeformConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=_same_padding(kernel_size),
            deform_groups=deform_groups,
            sampling_chunk_size=sampling_chunk_size,
            align_corners=align_corners,
        )
        self.activation = _activation(activation)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(input_tensor))


class SpatialUpsample3d(nn.Module):
    """Double H/W with resize-convolution, or reproduce archived transposed weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        *,
        mode: str = "nearest_conv",
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "nearest_conv":
            # Resize only space, then apply one shared stride-one kernel at
            # every location. Replicated padding also keeps a constant signal
            # constant at the spatial/temporal boundaries.
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                padding_mode="replicate",
            )
        elif mode == "transpose":
            # Keep both the math and state-dict names of the 400k baseline.
            self.upsample = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                output_padding=(0, 1, 1),
            )
        else:
            raise ValueError(f"unsupported registration upsample mode: {mode!r}")
        self.activation = _activation(activation)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.mode == "nearest_conv":
            resized = F.interpolate(input_tensor, scale_factor=(1, 2, 2), mode="nearest")
            return self.activation(self.conv(resized))
        return self.activation(self.upsample(input_tensor))


class FlowHead(nn.Module):
    """Predict one `(dx, dy)` backward flow for every input frame."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: tuple[int, int, int],
        reference_index: int,
        zero_reference_flow: bool,
    ) -> None:
        super().__init__()
        self.reference_index = reference_index
        self.zero_reference_flow = zero_reference_flow
        self.projection = nn.Conv3d(
            in_channels,
            2,
            kernel_size=kernel_size,
            padding=_same_padding(kernel_size),
        )
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flow = self.projection(features).permute(0, 2, 1, 3, 4).contiguous()
        if self.zero_reference_flow:
            mask = flow.new_ones((1, flow.shape[1], 1, 1, 1))
            mask[:, self.reference_index] = 0
            flow = flow * mask
        return flow


class DeTurbRegistration(nn.Module):
    """Depth-4 deformable 3D U-Net with three residual flow scales."""

    def __init__(
        self,
        contract: DeTurbContract | None = None,
        config: RegistrationConfig | None = None,
    ) -> None:
        super().__init__()
        self.contract = contract or DeTurbContract()
        self.config = config or RegistrationConfig(
            zero_reference_flow=not self.contract.returns_full_clip,
            post_warp_kernel=(3, 7, 7) if self.contract.returns_full_clip else None,
            upsample_mode="nearest_conv" if self.contract.returns_full_clip else "transpose",
        )
        if self.contract.returns_full_clip:
            if self.config.zero_reference_flow:
                raise ValueError("full_clip registration must correct every frame; disable zero_reference_flow")
            if self.config.post_warp_kernel is None:
                raise ValueError("deturb_v2_clip requires a post-warp RGB convolution")
        encoder = self.config.encoder_channels
        decoder = self.config.decoder_channels
        kernels = self.config.encoder_kernels

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.encoder_level1 = ConvActivation3d(
            self.contract.input_channels,
            encoder[0],
            kernels[0],
            self.config.activation,
        )
        self.encoder_level2 = DeformActivation3d(
            encoder[0],
            encoder[1],
            kernels[1],
            deform_groups=self.config.deform_groups[0],
            sampling_chunk_size=self.config.sampling_chunk_size,
            align_corners=self.config.align_corners,
            activation=self.config.activation,
        )
        self.encoder_level3 = DeformActivation3d(
            encoder[1],
            encoder[2],
            kernels[2],
            deform_groups=self.config.deform_groups[1],
            sampling_chunk_size=self.config.sampling_chunk_size,
            align_corners=self.config.align_corners,
            activation=self.config.activation,
        )
        self.bottleneck = DeformActivation3d(
            encoder[2],
            encoder[3],
            kernels[3],
            deform_groups=self.config.deform_groups[2],
            sampling_chunk_size=self.config.sampling_chunk_size,
            align_corners=self.config.align_corners,
            activation=self.config.activation,
        )

        self.up_coarse = SpatialUpsample3d(
            encoder[3], decoder[0], self.config.activation, mode=self.config.upsample_mode
        )
        self.decoder_coarse = ConvActivation3d(
            decoder[0] + encoder[2],
            decoder[0],
            self.config.decoder_kernel,
            self.config.activation,
        )
        self.up_mid = SpatialUpsample3d(
            decoder[0], decoder[1], self.config.activation, mode=self.config.upsample_mode
        )
        self.decoder_mid = ConvActivation3d(
            decoder[1] + encoder[1],
            decoder[1],
            self.config.decoder_kernel,
            self.config.activation,
        )
        self.up_full = SpatialUpsample3d(
            decoder[1], decoder[2], self.config.activation, mode=self.config.upsample_mode
        )
        self.decoder_full = ConvActivation3d(
            decoder[2] + encoder[0],
            decoder[2],
            self.config.decoder_kernel,
            self.config.activation,
        )

        flow_head_args = (
            self.config.flow_kernel,
            self.contract.reference_index,
            self.config.zero_reference_flow,
        )
        self.flow_head_coarse = FlowHead(decoder[0], *flow_head_args)
        self.flow_head_mid = FlowHead(decoder[1], *flow_head_args)
        self.flow_head_full = FlowHead(decoder[2], *flow_head_args)
        self.post_warp_conv = None
        if self.config.post_warp_kernel is not None:
            self.post_warp_conv = nn.Conv3d(
                self.contract.input_channels,
                self.contract.input_channels,
                kernel_size=self.config.post_warp_kernel,
                padding=_same_padding(self.config.post_warp_kernel),
            )
            # Start as an identity RGB refinement; learn spatial/temporal
            # corrections without initially destroying the warped signal.
            nn.init.dirac_(self.post_warp_conv.weight)
            nn.init.zeros_(self.post_warp_conv.bias)

    @staticmethod
    def _concat_skip(
        upsampled: torch.Tensor,
        skip: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if upsampled.shape[0] != skip.shape[0] or upsampled.shape[2:] != skip.shape[2:]:
            raise RuntimeError(
                f"{name} skip shape mismatch: {tuple(upsampled.shape)} vs "
                f"{tuple(skip.shape)}"
            )
        return torch.cat((upsampled, skip), dim=1)

    def _validate_input(self, clip: torch.Tensor) -> None:
        self.contract.validate_model_input(clip)
        height, width = clip.shape[-2:]
        divisor = self.config.spatial_divisor
        if height < divisor or width < divisor:
            raise ValueError(
                f"clip H/W must be at least {divisor} for three spatial pools"
            )
        if height % divisor or width % divisor:
            raise ValueError(
                f"clip H/W must be divisible by {divisor}, got {(height, width)}"
            )

    def forward(self, clip: torch.Tensor) -> RegistrationOutput:
        self._validate_input(clip)
        full_size = clip.shape[-2:]

        feature_level1 = self.encoder_level1(clip)
        feature_level2 = self.encoder_level2(self.pool(feature_level1))
        feature_level3 = self.encoder_level3(self.pool(feature_level2))
        base = self.bottleneck(self.pool(feature_level3))

        coarse_features = self.decoder_coarse(
            self._concat_skip(
                self.up_coarse(base),
                feature_level3,
                "coarse",
            )
        )
        mid_features = self.decoder_mid(
            self._concat_skip(
                self.up_mid(coarse_features),
                feature_level2,
                "mid",
            )
        )
        full_features = self.decoder_full(
            self._concat_skip(
                self.up_full(mid_features),
                feature_level1,
                "full",
            )
        )

        residual_coarse = self.flow_head_coarse(coarse_features)
        residual_mid = self.flow_head_mid(mid_features)
        residual_full = self.flow_head_full(full_features)

        cumulative_coarse = residual_coarse
        coarse_at_mid = resize_flow_2d(
            cumulative_coarse,
            residual_mid.shape[-2:],
            align_corners=self.config.align_corners,
        )
        cumulative_mid = compose_backward_flows(
            coarse_at_mid,
            residual_mid,
            padding_mode=self.config.flow_padding_mode,
            align_corners=self.config.align_corners,
        )
        mid_at_full = resize_flow_2d(
            cumulative_mid,
            full_size,
            align_corners=self.config.align_corners,
        )
        cumulative_full = compose_backward_flows(
            mid_at_full,
            residual_full,
            padding_mode=self.config.flow_padding_mode,
            align_corners=self.config.align_corners,
        )

        coarse_at_full = resize_flow_2d(
            cumulative_coarse,
            full_size,
            align_corners=self.config.align_corners,
        )
        aligned_coarse = warp_frames_2d(
            clip,
            coarse_at_full,
            padding_mode=self.config.flow_padding_mode,
            align_corners=self.config.align_corners,
        )
        aligned_mid = warp_frames_2d(
            clip,
            mid_at_full,
            padding_mode=self.config.flow_padding_mode,
            align_corners=self.config.align_corners,
        )
        warped_full = warp_frames_2d(
            clip,
            cumulative_full,
            padding_mode=self.config.flow_padding_mode,
            align_corners=self.config.align_corners,
        )
        aligned = (
            self.post_warp_conv(warped_full)
            if self.post_warp_conv is not None else warped_full
        )

        return RegistrationOutput(
            aligned=aligned,
            aligned_coarse=aligned_coarse,
            aligned_mid=aligned_mid,
            residual_flow_coarse=residual_coarse,
            residual_flow_mid=residual_mid,
            residual_flow_full=residual_full,
            cumulative_flow_coarse=cumulative_coarse,
            cumulative_flow_mid=cumulative_mid,
            cumulative_flow_full=cumulative_full,
            warped_full=warped_full,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
