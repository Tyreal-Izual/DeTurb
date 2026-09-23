"""Final public model API for DeTurb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import (
    DeTurbContract,
    FusionConfig,
    RegistrationConfig,
    load_contract_config,
    load_fusion_config,
    load_registration_config,
)
from .fusion_swin3d import DeTurbFusion
from .registration import DeTurbRegistration
from .types import DeTurbOutput


class DeTurb(nn.Module):
    """Non-rigid registration followed by 3D Swin feature fusion."""

    def __init__(
        self,
        contract: DeTurbContract | None = None,
        registration_config: RegistrationConfig | None = None,
        fusion_config: FusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.contract = contract or DeTurbContract()
        self.fusion_config = fusion_config or FusionConfig(
            attention_mask_mode="strict" if self.contract.returns_full_clip else "legacy"
        )
        self.registration = DeTurbRegistration(
            self.contract,
            registration_config,
        )
        self.registration_config = self.registration.config
        self.fusion = DeTurbFusion(self.contract, self.fusion_config)

    @classmethod
    def from_config(cls, path: str | Path) -> "DeTurb":
        return cls(
            contract=load_contract_config(path),
            registration_config=load_registration_config(path),
            fusion_config=load_fusion_config(path),
        )

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "DeTurb":
        contract = DeTurbContract.from_mapping(spec["contract"])
        if spec.get("model_name", contract.model_id) != contract.model_id:
            raise ValueError("Model spec identity and contract disagree")
        return cls(contract, RegistrationConfig.from_mapping(spec["registration"]),
                   FusionConfig.from_mapping(spec["fusion"]))

    def forward(
        self,
        clip: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | DeTurbOutput:
        registration_output = self.registration(clip)
        fusion_output = self.fusion(registration_output.aligned)
        if not return_aux:
            return fusion_output.restored
        return DeTurbOutput(
            restored=fusion_output.restored,
            registration=registration_output,
            fusion=fusion_output,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def checkpoint_version(self) -> int:
        revised = (
            self.registration_config.post_warp_kernel is not None
            or self.registration_config.upsample_mode != "transpose"
            or self.fusion_config.attention_mask_mode != "legacy"
        )
        return 6 if revised else 5

    def model_spec(self) -> dict[str, Any]:
        return {
            "model_name": self.contract.model_id,
            "model_version": (4 if self.fusion_config.attention_mask_mode == "strict"
                              else 3 if self.checkpoint_version == 6 else 2),
            "contract": self.contract.as_metadata(),
            "registration": self.registration_config.as_metadata(),
            "fusion": self.fusion_config.as_metadata(),
            "parameters": {
                "registration": self.registration.parameter_count(),
                "fusion": self.fusion.parameter_count(),
                "total": self.parameter_count(),
            },
        }
