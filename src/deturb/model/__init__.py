"""Paper-aligned DeTurb building blocks."""

from .config import (
    DeTurbContract,
    DeTurbLossConfig,
    FusionConfig,
    RegistrationConfig,
    load_contract_config,
    load_fusion_config,
    load_loss_config,
    load_registration_config,
)
from .deform_conv3d import (
    DeformConv3d,
    DeformConv3dReference,
    deform_conv3d,
    deform_conv3d_reference,
)
from .temporal import (
    build_reference_windows,
    dataset_clip_to_model_input,
    materialize_model_window,
    reference_window_for_output,
    select_reference_target,
    select_output_target,
)
from .registration import DeTurbRegistration
from .fusion_swin3d import (
    DeTurbFusion,
    SwinTransformerBlock3D,
    WindowAttention3D,
)
from .losses import (
    CharbonnierLoss,
    DeTurbLoss,
    LaplacianEdgeLoss,
    LossBreakdown,
)
from .model import DeTurb
from .types import (
    DeTurbOutput,
    FusionOutput,
    RegistrationOutput,
    TemporalReferenceWindow,
)
from .warping import compose_backward_flows, resize_flow_2d, warp_frames_2d

__all__ = [
    "DeTurbContract",
    "DeTurb",
    "DeTurbLoss",
    "DeTurbLossConfig",
    "DeTurbOutput",
    "DeTurbRegistration",
    "DeTurbFusion",
    "DeformConv3d",
    "DeformConv3dReference",
    "FusionConfig",
    "FusionOutput",
    "CharbonnierLoss",
    "LaplacianEdgeLoss",
    "LossBreakdown",
    "RegistrationConfig",
    "RegistrationOutput",
    "TemporalReferenceWindow",
    "build_reference_windows",
    "dataset_clip_to_model_input",
    "deform_conv3d",
    "deform_conv3d_reference",
    "materialize_model_window",
    "load_contract_config",
    "load_fusion_config",
    "load_loss_config",
    "load_registration_config",
    "reference_window_for_output",
    "select_reference_target",
    "select_output_target",
    "compose_backward_flows",
    "resize_flow_2d",
    "warp_frames_2d",
    "SwinTransformerBlock3D",
    "WindowAttention3D",
]
