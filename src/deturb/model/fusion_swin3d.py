"""3D shifted-window feature-fusion U-Net for DeTurb."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeTurbContract, FusionConfig
from .types import FusionOutput


Triple = tuple[int, int, int]


def _window_partition(x: torch.Tensor, window_size: Triple) -> torch.Tensor:
    """Partition channel-last `[B,T,H,W,C]` into `[B*nW,N,C]`."""

    batch, frames, height, width, channels = x.shape
    window_t, window_h, window_w = window_size
    return (
        x.view(
            batch,
            frames // window_t,
            window_t,
            height // window_h,
            window_h,
            width // window_w,
            window_w,
            channels,
        )
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(-1, window_t * window_h * window_w, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window_size: Triple,
    batch: int,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse `_window_partition` into channel-last padded features."""

    window_t, window_h, window_w = window_size
    channels = windows.shape[-1]
    return (
        windows.view(
            batch,
            frames // window_t,
            height // window_h,
            width // window_w,
            window_t,
            window_h,
            window_w,
            channels,
        )
        .permute(0, 1, 4, 2, 5, 3, 6, 7)
        .reshape(batch, frames, height, width, channels)
    )


def _axis_slices(size: int, window: int, shift: int) -> tuple[slice, ...]:
    if shift == 0:
        return (slice(0, size),)
    return (
        slice(0, -window),
        slice(-window, -shift),
        slice(-shift, None),
    )


def _shift_attention_mask(
    padded_shape: Triple,
    window_size: Triple,
    shift_size: Triple,
    device: torch.device,
) -> torch.Tensor | None:
    if not any(shift_size):
        return None
    frames, height, width = padded_shape
    mask = torch.zeros((1, frames, height, width, 1), device=device)
    region = 0
    for frame_slice in _axis_slices(frames, window_size[0], shift_size[0]):
        for height_slice in _axis_slices(height, window_size[1], shift_size[1]):
            for width_slice in _axis_slices(width, window_size[2], shift_size[2]):
                mask[:, frame_slice, height_slice, width_slice, :] = region
                region += 1
    mask_windows = _window_partition(mask, window_size).squeeze(-1)
    difference = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return difference.ne(0)


def _relative_position_index(
    effective_window: Triple,
    configured_window: Triple,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(effective_window[0], device=device),
            torch.arange(effective_window[1], device=device),
            torch.arange(effective_window[2], device=device),
            indexing="ij",
        )
    ).flatten(1)
    relative = coordinates[:, :, None] - coordinates[:, None, :]
    relative[0] += configured_window[0] - 1
    relative[1] += configured_window[1] - 1
    relative[2] += configured_window[2] - 1
    relative[0] *= (2 * configured_window[1] - 1) * (
        2 * configured_window[2] - 1
    )
    relative[1] *= 2 * configured_window[2] - 1
    return relative.sum(0).long()


class WindowAttention3D(nn.Module):
    """Dynamic padded regular or shifted 3D window self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Triple,
        *,
        shifted: bool,
        qkv_bias: bool,
        relative_position_bias: bool,
        attention_dropout: float,
        projection_dropout: float,
        mask_mode: str = "strict",
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        if mask_mode not in ("legacy", "strict"):
            raise ValueError("mask_mode must be legacy or strict")
        self.mask_mode = mask_mode
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)
        if relative_position_bias:
            table_size = math.prod(2 * size - 1 for size in window_size)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(table_size, num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.register_parameter("relative_position_bias_table", None)

    def _effective_window_and_shift(self, shape: Triple) -> tuple[Triple, Triple]:
        effective_window = tuple(
            min(configured, current)
            for configured, current in zip(self.window_size, shape)
        )
        padded_shape = tuple(
            math.ceil(current / window) * window
            for current, window in zip(shape, effective_window)
        )
        shift_size = tuple(
            window // 2 if self.shifted and padded > window else 0
            for window, padded in zip(effective_window, padded_shape)
        )
        return effective_window, shift_size  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[-1] != self.dim:
            raise ValueError("attention input must have shape [B,T,H,W,C]")
        batch, frames, height, width, channels = x.shape
        original_shape = (frames, height, width)
        window_size, shift_size = self._effective_window_and_shift(original_shape)
        padded_shape = tuple(
            math.ceil(current / window) * window
            for current, window in zip(original_shape, window_size)
        )
        pad_t = padded_shape[0] - frames
        pad_h = padded_shape[1] - height
        pad_w = padded_shape[2] - width
        channel_first = x.permute(0, 4, 1, 2, 3)
        channel_first = F.pad(channel_first, (0, pad_w, 0, pad_h, 0, pad_t))
        x = channel_first.permute(0, 2, 3, 4, 1)
        validity = torch.ones(
            (1, 1, frames, height, width),
            dtype=x.dtype,
            device=x.device,
        )
        validity = F.pad(validity, (0, pad_w, 0, pad_h, 0, pad_t))
        validity = validity.permute(0, 2, 3, 4, 1)

        if any(shift_size):
            shifts = tuple(-shift for shift in shift_size)
            x = torch.roll(x, shifts=shifts, dims=(1, 2, 3))
            validity = torch.roll(validity, shifts=shifts, dims=(1, 2, 3))

        windows = _window_partition(x, window_size)
        valid_windows = _window_partition(validity, window_size).squeeze(-1).bool()
        tokens = windows.shape[1]
        qkv = (
            self.qkv(windows)
            .reshape(-1, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_index = _relative_position_index(
                window_size,
                self.window_size,
                x.device,
            )
            relative_bias = self.relative_position_bias_table[
                relative_index.reshape(-1)
            ].reshape(tokens, tokens, self.num_heads)
            attention = attention + relative_bias.permute(2, 0, 1).unsqueeze(0)

        windows_per_sample = math.prod(
            padded // window
            for padded, window in zip(padded_shape, window_size)
        )
        shift_mask = _shift_attention_mask(
            padded_shape,
            window_size,
            shift_size,
            x.device,
        )
        invalid_keys = ~valid_windows
        combined_mask = invalid_keys.unsqueeze(1).expand(-1, tokens, -1)
        if shift_mask is not None:
            combined_mask = combined_mask | shift_mask
        attention = attention.view(
            batch,
            windows_per_sample,
            self.num_heads,
            tokens,
            tokens,
        )
        mask = combined_mask.view(1, windows_per_sample, 1, tokens, tokens)
        if self.mask_mode == "legacy":
            attention = attention.masked_fill(mask, -100.0).view(-1, self.num_heads, tokens, tokens).softmax(dim=-1)
        else:
            # Padded queries can have no valid keys in their shifted region.
            # Avoid softmax(all -inf), then explicitly zero masked probabilities.
            attention = attention.masked_fill(mask, -torch.inf)
            attention = attention.masked_fill(mask.all(dim=-1, keepdim=True), 0.0)
            attention = attention.softmax(dim=-1).masked_fill(mask, 0.0)
        attention = self.attention_dropout(attention.view(-1, self.num_heads, tokens, tokens))
        output = (attention @ value).transpose(1, 2).reshape(-1, tokens, channels)
        output = self.projection_dropout(self.projection(output))
        output = _window_reverse(
            output,
            window_size,
            batch,
            *padded_shape,
        )
        if any(shift_size):
            output = torch.roll(output, shifts=shift_size, dims=(1, 2, 3))
        return output[:, :frames, :height, :width, :].contiguous()


class SwinTransformerBlock3D(nn.Module):
    """Pre-norm 3D Swin attention and MLP residual block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Triple,
        *,
        shifted: bool,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        qkv_bias: bool,
        relative_position_bias: bool,
        mask_mode: str = "strict",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention3D(
            dim,
            num_heads,
            window_size,
            shifted=shifted,
            qkv_bias=qkv_bias,
            relative_position_bias=relative_position_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            mask_mode=mask_mode,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Swin block input must have shape [B,C,T,H,W]")
        channel_last = x.permute(0, 2, 3, 4, 1)
        channel_last = channel_last + self.attention(self.norm1(channel_last))
        channel_last = channel_last + self.mlp(self.norm2(channel_last))
        return channel_last.permute(0, 4, 1, 2, 3).contiguous()


class ChannelLayerNorm3D(nn.Module):
    """Apply LayerNorm over channels without fixing T/H/W."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_last = x.permute(0, 2, 3, 4, 1)
        return self.norm(channel_last).permute(0, 4, 1, 2, 3).contiguous()


class ResidualConvBranch3D(nn.Module):
    """Two-convolution pre-norm 3D ResNet basic block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm3D(channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = ChannelLayerNorm3D(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(self.norm1(x)))
        x = self.conv2(self.norm2(x))
        return residual + x


class HybridSwinStage3D(nn.Module):
    """Sum local residual-convolution and global 3D Swin branches."""

    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        config: FusionConfig,
    ) -> None:
        super().__init__()
        self.conv_branch = ResidualConvBranch3D(channels)
        self.swin_blocks = nn.Sequential(
            *(
                SwinTransformerBlock3D(
                    channels,
                    num_heads,
                    config.window_size,
                    shifted=bool(index % 2),
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    qkv_bias=config.qkv_bias,
                    relative_position_bias=config.relative_position_bias,
                    mask_mode=config.attention_mask_mode,
                )
                for index in range(depth)
            )
        )
        self.output_norm = ChannelLayerNorm3D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.conv_branch(x) + self.swin_blocks(x))


class SamePadConv3D(nn.Module):
    """Stride-one Conv3d with explicit asymmetric padding for even kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Triple,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pads = []
        for size in reversed(self.kernel_size):
            total = size - 1
            pads.extend((total // 2, total - total // 2))
        return self.conv(F.pad(x, tuple(pads)))


class FusionStem3D(nn.Module):
    """Two 3D convolutions producing 32 channels at half spatial resolution."""

    def __init__(self, input_channels: int, stem_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv3d(
            input_channels,
            stem_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.refine = SamePadConv3D(
            stem_channels,
            stem_channels,
            kernel_size=(3, 4, 4),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.refine(self.activation(self.downsample(x))))


class SpatialDownsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SpatialUpsample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
    ) -> None:
        super().__init__()
        self.projection = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, factor, factor),
            stride=(1, factor, factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class EncoderStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        heads: int,
        config: FusionConfig,
    ) -> None:
        super().__init__()
        self.downsample = SpatialDownsample3D(in_channels, out_channels)
        self.hybrid = HybridSwinStage3D(out_channels, depth, heads, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hybrid(self.downsample(x))


class DecoderStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        heads: int,
        config: FusionConfig,
    ) -> None:
        super().__init__()
        self.upsample = SpatialUpsample3D(in_channels, out_channels)
        self.hybrid = HybridSwinStage3D(out_channels, depth, heads, config)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape != skip.shape:
            raise RuntimeError(
                f"fusion skip mismatch: {tuple(x.shape)} vs {tuple(skip.shape)}"
            )
        return self.hybrid(x + skip)


class DeTurbFusion(nn.Module):
    """Depth-4 3D Swin U-Net restoring a reference frame or a time-matched clip."""

    def __init__(
        self,
        contract: DeTurbContract | None = None,
        config: FusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.contract = contract or DeTurbContract()
        self.config = config or FusionConfig(
            attention_mask_mode="strict" if self.contract.returns_full_clip else "legacy"
        )
        encoder = self.config.encoder_channels
        decoder = self.config.decoder_channels
        self.stem = FusionStem3D(
            self.contract.input_channels,
            self.config.stem_channels,
        )
        self.encoder_stages = nn.ModuleList(
            (
                EncoderStage3D(
                    self.config.stem_channels,
                    encoder[0],
                    self.config.encoder_depths[0],
                    self.config.encoder_heads[0],
                    self.config,
                ),
                EncoderStage3D(
                    encoder[0],
                    encoder[1],
                    self.config.encoder_depths[1],
                    self.config.encoder_heads[1],
                    self.config,
                ),
                EncoderStage3D(
                    encoder[1],
                    encoder[2],
                    self.config.encoder_depths[2],
                    self.config.encoder_heads[2],
                    self.config,
                ),
                EncoderStage3D(
                    encoder[2],
                    encoder[3],
                    self.config.encoder_depths[3],
                    self.config.encoder_heads[3],
                    self.config,
                ),
            )
        )
        self.decoder_stages = nn.ModuleList(
            (
                DecoderStage3D(
                    encoder[3],
                    decoder[0],
                    self.config.decoder_depths[0],
                    self.config.decoder_heads[0],
                    self.config,
                ),
                DecoderStage3D(
                    decoder[0],
                    decoder[1],
                    self.config.decoder_depths[1],
                    self.config.decoder_heads[1],
                    self.config,
                ),
                DecoderStage3D(
                    decoder[1],
                    decoder[2],
                    self.config.decoder_depths[2],
                    self.config.decoder_heads[2],
                    self.config,
                ),
            )
        )
        self.final_upsample = SpatialUpsample3D(
            decoder[2],
            self.config.stem_channels,
            factor=4,
        )
        self.output_projection = nn.Conv3d(
            self.config.stem_channels,
            self.contract.output_channels,
            kernel_size=1,
        )

    def _pad_spatial(self, clip: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = clip.shape[-2:]
        divisor = self.config.spatial_divisor
        padded_height = math.ceil(height / divisor) * divisor
        padded_width = math.ceil(width / divisor) * divisor
        pad_height = padded_height - height
        pad_width = padded_width - width
        if pad_height == 0 and pad_width == 0:
            return clip, height, width
        mode = self.config.spatial_padding_mode
        if mode == "reflect" and (pad_height >= height or pad_width >= width):
            raise ValueError(
                "reflect fusion padding requires each pad to be smaller than input"
            )
        if mode == "zeros":
            padded = F.pad(clip, (0, pad_width, 0, pad_height, 0, 0))
        else:
            padded = F.pad(
                clip,
                (0, pad_width, 0, pad_height, 0, 0),
                mode=mode,
            )
        return padded, height, width

    def forward(self, aligned_clip: torch.Tensor) -> FusionOutput:
        self.contract.validate_model_input(aligned_clip, name="aligned_clip")
        padded, original_height, original_width = self._pad_spatial(aligned_clip)
        stem = self.stem(padded)
        encoder1 = self.encoder_stages[0](stem)
        encoder2 = self.encoder_stages[1](encoder1)
        encoder3 = self.encoder_stages[2](encoder2)
        bottleneck = self.encoder_stages[3](encoder3)
        decoder3 = self.decoder_stages[0](bottleneck, encoder3)
        decoder2 = self.decoder_stages[1](decoder3, encoder2)
        decoder1 = self.decoder_stages[2](decoder2, encoder1)
        full_features = self.final_upsample(decoder1)
        predicted_clip = self.output_projection(full_features)
        predicted_clip = predicted_clip[
            ...,
            :original_height,
            :original_width,
        ].contiguous()
        restored = (
            predicted_clip if self.contract.returns_full_clip
            else predicted_clip[:, :, self.contract.reference_index].contiguous()
        )
        if self.config.output_residual:
            restored = restored + (
                aligned_clip if self.contract.returns_full_clip
                else self.contract.select_reference(aligned_clip)
            )
        return FusionOutput(
            restored=restored,
            predicted_clip=predicted_clip,
            padded_spatial_shape=tuple(padded.shape[-2:]),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
