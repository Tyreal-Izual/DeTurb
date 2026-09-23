"""Explicit input/output contract for the paper-aligned DeTurb model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


_TENSOR_LAYOUT = "BCTHW"
_OUTPUT_MODE = "reference_frame"
_MODEL_ID = "deturb_v2_paper"
_MODEL_OUTPUT_MODES = {_MODEL_ID: _OUTPUT_MODE, "deturb_v2_clip": "full_clip"}
_SUPPORTED_TEMPORAL_PADDING = {"replicate"}
_SUPPORTED_FLOW_PADDING = {"zeros", "border", "reflection"}


@dataclass(frozen=True, slots=True)
class DeTurbContract:
    """Model and temporal-window contract shared by data, train and inference.

    The model consumes floating-point clips in ``[B, C, T, H, W]`` layout and
    restores either a reference image (the archived model) or a full BCTHW clip.
    """

    model_id: str = _MODEL_ID
    input_frames: int = 12
    reference_index: int = 6
    input_channels: int = 3
    output_channels: int = 3
    tensor_layout: str = _TENSOR_LAYOUT
    output_mode: str = _OUTPUT_MODE
    temporal_padding: str = "replicate"

    def __post_init__(self) -> None:
        if self.model_id not in _MODEL_OUTPUT_MODES:
            raise ValueError(f"unsupported model_id: {self.model_id!r}")
        if self.input_frames <= 0:
            raise ValueError("input_frames must be positive")
        if not 0 <= self.reference_index < self.input_frames:
            raise ValueError(
                "reference_index must satisfy 0 <= reference_index < input_frames"
            )
        if self.input_channels <= 0 or self.output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        if self.tensor_layout != _TENSOR_LAYOUT:
            raise ValueError(f"tensor_layout must be {_TENSOR_LAYOUT!r}")
        expected_mode = _MODEL_OUTPUT_MODES[self.model_id]
        if self.output_mode != expected_mode:
            raise ValueError(f"{self.model_id} requires output_mode={expected_mode!r}")
        if self.temporal_padding not in _SUPPORTED_TEMPORAL_PADDING:
            raise ValueError(
                "temporal_padding must be one of "
                f"{sorted(_SUPPORTED_TEMPORAL_PADDING)}"
            )

    @property
    def returns_full_clip(self) -> bool:
        return self.output_mode == "full_clip"

    @property
    def output_frames(self) -> int:
        return self.input_frames if self.returns_full_clip else 1

    @property
    def left_context(self) -> int:
        return self.reference_index

    @property
    def right_context(self) -> int:
        return self.input_frames - self.reference_index - 1

    def validate_model_input(
        self,
        clip: torch.Tensor,
        *,
        name: str = "clip",
    ) -> None:
        if not isinstance(clip, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if clip.ndim != 5:
            raise ValueError(f"{name} must have shape [B, C, T, H, W]")
        batch, channels, frames, height, width = clip.shape
        if batch <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"{name} dimensions must be positive")
        if channels != self.input_channels:
            raise ValueError(
                f"{name} has {channels} channels; expected {self.input_channels}"
            )
        if frames != self.input_frames:
            raise ValueError(
                f"{name} has {frames} frames; expected {self.input_frames}"
            )
        if not torch.is_floating_point(clip):
            raise TypeError(f"{name} must use a floating-point dtype")

    def select_reference(self, clip: torch.Tensor) -> torch.Tensor:
        """Select ``[B, C, H, W]`` target/output frame from a model clip."""

        self.validate_model_input(clip)
        return clip[:, :, self.reference_index, :, :].contiguous()

    def output_reference(self, output: torch.Tensor) -> torch.Tensor:
        """Select a preview/reference metric image without discarding training targets."""
        expected_ndim = 5 if self.returns_full_clip else 4
        if output.ndim != expected_ndim or output.shape[1] != self.output_channels:
            raise ValueError("output does not match the model output contract")
        if self.returns_full_clip:
            if output.shape[2] != self.input_frames:
                raise ValueError("output clip has the wrong number of frames")
            return output[:, :, self.reference_index].contiguous()
        return output

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DeTurbContract":
        if not isinstance(values, dict):
            raise TypeError("contract values must be a dictionary")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid DeTurb contract fields: {error}") from error


def _positive_int_tuple(
    value: Any,
    length: int,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} integers")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value
    ):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(value)


def _odd_kernel(value: Any, name: str) -> tuple[int, int, int]:
    kernel = _positive_int_tuple(value, 3, name)
    if any(item % 2 == 0 for item in kernel):
        raise ValueError(f"{name} values must be odd")
    return kernel  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RegistrationConfig:
    """Architecture and warping policy for the DeTurb non-rigid registration net."""

    encoder_channels: tuple[int, int, int, int] = (64, 256, 256, 512)
    decoder_channels: tuple[int, int, int] = (256, 128, 64)
    encoder_kernels: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 3, 3))
    decoder_kernel: tuple[int, int, int] = (3, 3, 3)
    flow_kernel: tuple[int, int, int] = (3, 3, 3)
    deform_groups: tuple[int, int, int] = (1, 1, 1)
    sampling_chunk_size: int = 16
    activation: str = "leaky_relu"
    flow_padding_mode: str = "border"
    align_corners: bool = False
    zero_reference_flow: bool = True
    # None preserves the archived 400k architecture. The clip variant enables
    # a temporal-size-3, spatial-size-7 RGB convolution after the final warp.
    post_warp_kernel: tuple[int, int, int] | None = None
    # Historical configs omitted this field and used transposed convolutions.
    upsample_mode: str = "transpose"

    def __post_init__(self) -> None:
        encoder_channels = _positive_int_tuple(
            self.encoder_channels,
            4,
            "encoder_channels",
        )
        decoder_channels = _positive_int_tuple(
            self.decoder_channels,
            3,
            "decoder_channels",
        )
        if not isinstance(self.encoder_kernels, (list, tuple)) or len(
            self.encoder_kernels
        ) != 4:
            raise ValueError("encoder_kernels must contain four 3D kernels")
        encoder_kernels = tuple(
            _odd_kernel(kernel, f"encoder_kernels[{index}]")
            for index, kernel in enumerate(self.encoder_kernels)
        )
        decoder_kernel = _odd_kernel(self.decoder_kernel, "decoder_kernel")
        flow_kernel = _odd_kernel(self.flow_kernel, "flow_kernel")
        deform_groups = _positive_int_tuple(
            self.deform_groups,
            3,
            "deform_groups",
        )
        deform_input_channels = encoder_channels[:3]
        if any(
            channels % groups
            for channels, groups in zip(deform_input_channels, deform_groups)
        ):
            raise ValueError(
                "each deform_groups value must divide its stage input channels"
            )
        if (
            not isinstance(self.sampling_chunk_size, int)
            or isinstance(self.sampling_chunk_size, bool)
            or self.sampling_chunk_size <= 0
        ):
            raise ValueError("sampling_chunk_size must be a positive integer")
        if self.activation not in {"relu", "leaky_relu", "gelu"}:
            raise ValueError("activation must be relu, leaky_relu or gelu")
        if self.upsample_mode not in ("transpose", "nearest_conv"):
            raise ValueError("upsample_mode must be transpose or nearest_conv")
        if self.flow_padding_mode not in _SUPPORTED_FLOW_PADDING:
            raise ValueError(
                "flow_padding_mode must be one of "
                f"{sorted(_SUPPORTED_FLOW_PADDING)}"
            )
        if not isinstance(self.align_corners, bool):
            raise TypeError("align_corners must be bool")
        if not isinstance(self.zero_reference_flow, bool):
            raise TypeError("zero_reference_flow must be bool")
        if self.post_warp_kernel is not None:
            object.__setattr__(
                self, "post_warp_kernel", _odd_kernel(self.post_warp_kernel, "post_warp_kernel")
            )

        object.__setattr__(self, "encoder_channels", encoder_channels)
        object.__setattr__(self, "decoder_channels", decoder_channels)
        object.__setattr__(self, "encoder_kernels", encoder_kernels)
        object.__setattr__(self, "decoder_kernel", decoder_kernel)
        object.__setattr__(self, "flow_kernel", flow_kernel)
        object.__setattr__(self, "deform_groups", deform_groups)

    @property
    def spatial_divisor(self) -> int:
        return 8

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "RegistrationConfig":
        if not isinstance(values, dict):
            raise TypeError("registration values must be a dictionary")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid registration config fields: {error}") from error


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Architecture policy for the DeTurb 3D Swin feature-fusion U-Net."""

    stem_channels: int = 32
    encoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    decoder_channels: tuple[int, int, int] = (256, 128, 64)
    encoder_depths: tuple[int, int, int, int] = (2, 2, 6, 2)
    decoder_depths: tuple[int, int, int] = (6, 2, 2)
    encoder_heads: tuple[int, int, int, int] = (2, 4, 8, 16)
    decoder_heads: tuple[int, int, int] = (8, 4, 2)
    window_size: tuple[int, int, int] = (3, 8, 8)
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    qkv_bias: bool = True
    relative_position_bias: bool = True
    spatial_padding_mode: str = "replicate"
    output_residual: bool = False
    attention_mask_mode: str = "legacy"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stem_channels, int)
            or isinstance(self.stem_channels, bool)
            or self.stem_channels <= 0
        ):
            raise ValueError("stem_channels must be a positive integer")
        if self.attention_mask_mode not in ("legacy", "strict"):
            raise ValueError("attention_mask_mode must be legacy or strict")
        encoder_channels = _positive_int_tuple(
            self.encoder_channels,
            4,
            "encoder_channels",
        )
        decoder_channels = _positive_int_tuple(
            self.decoder_channels,
            3,
            "decoder_channels",
        )
        if decoder_channels != (
            encoder_channels[2],
            encoder_channels[1],
            encoder_channels[0],
        ):
            raise ValueError(
                "decoder_channels must match reversed encoder skip widths"
            )
        encoder_depths = _positive_int_tuple(
            self.encoder_depths,
            4,
            "encoder_depths",
        )
        decoder_depths = _positive_int_tuple(
            self.decoder_depths,
            3,
            "decoder_depths",
        )
        if any(depth % 2 for depth in encoder_depths + decoder_depths):
            raise ValueError("all Swin stage depths must be even")
        encoder_heads = _positive_int_tuple(
            self.encoder_heads,
            4,
            "encoder_heads",
        )
        decoder_heads = _positive_int_tuple(
            self.decoder_heads,
            3,
            "decoder_heads",
        )
        if any(
            channels % heads
            for channels, heads in zip(encoder_channels, encoder_heads)
        ) or any(
            channels % heads
            for channels, heads in zip(decoder_channels, decoder_heads)
        ):
            raise ValueError("every fusion width must be divisible by its heads")
        window_size = _positive_int_tuple(self.window_size, 3, "window_size")
        if (
            not isinstance(self.mlp_ratio, (int, float))
            or isinstance(self.mlp_ratio, bool)
            or self.mlp_ratio <= 0
        ):
            raise ValueError("mlp_ratio must be positive")
        for name, value in (
            ("dropout", self.dropout),
            ("attention_dropout", self.attention_dropout),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= value < 1
            ):
                raise ValueError(f"{name} must satisfy 0 <= value < 1")
        for name, value in (
            ("qkv_bias", self.qkv_bias),
            ("relative_position_bias", self.relative_position_bias),
            ("output_residual", self.output_residual),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if self.spatial_padding_mode not in {"zeros", "replicate", "reflect"}:
            raise ValueError(
                "spatial_padding_mode must be zeros, replicate or reflect"
            )

        object.__setattr__(self, "encoder_channels", encoder_channels)
        object.__setattr__(self, "decoder_channels", decoder_channels)
        object.__setattr__(self, "encoder_depths", encoder_depths)
        object.__setattr__(self, "decoder_depths", decoder_depths)
        object.__setattr__(self, "encoder_heads", encoder_heads)
        object.__setattr__(self, "decoder_heads", decoder_heads)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "mlp_ratio", float(self.mlp_ratio))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "attention_dropout", float(self.attention_dropout))

    @property
    def spatial_divisor(self) -> int:
        return 32

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "FusionConfig":
        if not isinstance(values, dict):
            raise TypeError("fusion values must be a dictionary")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid fusion config fields: {error}") from error


@dataclass(frozen=True, slots=True)
class DeTurbLossConfig:
    """Weights and schedule for joint DeTurb supervision."""

    charbonnier_epsilon: float = 1e-3
    final_weight: float = 1.0
    alignment_weight: float = 1.0
    alignment_stage_weights: tuple[float, float, float] = (0.6, 0.3, 0.1)
    edge_weight: float = 0.05
    edge_start_iteration: int = 300000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.charbonnier_epsilon, (int, float))
            or isinstance(self.charbonnier_epsilon, bool)
            or self.charbonnier_epsilon <= 0
        ):
            raise ValueError("charbonnier_epsilon must be positive")
        for name, value in (
            ("final_weight", self.final_weight),
            ("alignment_weight", self.alignment_weight),
            ("edge_weight", self.edge_weight),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")
        if self.final_weight <= 0:
            raise ValueError("final_weight must be positive")
        if (
            not isinstance(self.alignment_stage_weights, (list, tuple))
            or len(self.alignment_stage_weights) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
                for value in self.alignment_stage_weights
            )
        ):
            raise ValueError(
                "alignment_stage_weights must contain three non-negative values"
            )
        if self.alignment_weight > 0 and sum(self.alignment_stage_weights) <= 0:
            raise ValueError("alignment stage weights must have a positive sum")
        if (
            not isinstance(self.edge_start_iteration, int)
            or isinstance(self.edge_start_iteration, bool)
            or self.edge_start_iteration < 0
        ):
            raise ValueError("edge_start_iteration must be a non-negative integer")
        object.__setattr__(
            self,
            "alignment_stage_weights",
            tuple(float(value) for value in self.alignment_stage_weights),
        )
        for name in (
            "charbonnier_epsilon",
            "final_weight",
            "alignment_weight",
            "edge_weight",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DeTurbLossConfig":
        if not isinstance(values, dict):
            raise TypeError("loss values must be a dictionary")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid loss config fields: {error}") from error


def _load_config_payload(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read DeTurb config {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("DeTurb config root must be a JSON object")
    return config_path, payload


def load_contract_config(path: str | Path) -> DeTurbContract:
    """Load ``model.contract`` from a DeTurb JSON configuration."""

    _, payload = _load_config_payload(path)
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("DeTurb config must contain a model object")
    contract = model.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("DeTurb config must contain model.contract")
    return DeTurbContract.from_mapping(contract)


def load_registration_config(path: str | Path) -> RegistrationConfig:
    """Load ``model.registration`` from a DeTurb JSON configuration."""

    _, payload = _load_config_payload(path)
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("DeTurb config must contain a model object")
    registration = model.get("registration")
    if not isinstance(registration, dict):
        raise ValueError("DeTurb config must contain model.registration")
    return RegistrationConfig.from_mapping(registration)


def load_fusion_config(path: str | Path) -> FusionConfig:
    """Load ``model.fusion`` from a DeTurb JSON configuration."""

    _, payload = _load_config_payload(path)
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("DeTurb config must contain a model object")
    fusion = model.get("fusion")
    if not isinstance(fusion, dict):
        raise ValueError("DeTurb config must contain model.fusion")
    return FusionConfig.from_mapping(fusion)


def load_loss_config(path: str | Path) -> DeTurbLossConfig:
    """Load top-level ``loss`` from a DeTurb JSON configuration."""

    _, payload = _load_config_payload(path)
    loss = payload.get("loss")
    if not isinstance(loss, dict):
        raise ValueError("DeTurb config must contain a loss object")
    return DeTurbLossConfig.from_mapping(loss)
