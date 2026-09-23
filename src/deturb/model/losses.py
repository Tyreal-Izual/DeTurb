"""Joint reconstruction, alignment and edge losses for DeTurb."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeTurbLossConfig
from .types import DeTurbOutput


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("Charbonnier prediction and target shapes must match")
        return torch.sqrt(
            (prediction - target).square() + self.epsilon * self.epsilon
        ).mean()


class LaplacianEdgeLoss(nn.Module):
    """Laplacian-pyramid edge loss with a device-safe Gaussian buffer."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        kernel_1d = torch.tensor((0.05, 0.25, 0.4, 0.25, 0.05))
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("kernel", kernel_2d.view(1, 1, 5, 5))
        self.charbonnier = CharbonnierLoss(epsilon)

    def _gaussian(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("edge loss expects [B,C,H,W] tensors")
        channels = image.shape[1]
        kernel = self.kernel.to(dtype=image.dtype).expand(channels, 1, 5, 5)
        padded = F.pad(image, (2, 2, 2, 2), mode="replicate")
        return F.conv2d(padded, kernel, groups=channels)

    def _laplacian(self, image: torch.Tensor) -> torch.Tensor:
        filtered = self._gaussian(image)
        downsampled = filtered[:, :, ::2, ::2]
        upsampled = torch.zeros_like(filtered)
        upsampled[:, :, ::2, ::2] = downsampled * 4
        return image - self._gaussian(upsampled)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("edge prediction and target shapes must match")
        return self.charbonnier(
            self._laplacian(prediction),
            self._laplacian(target),
        )


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: torch.Tensor
    final: torch.Tensor
    alignment_full: torch.Tensor
    alignment_mid: torch.Tensor
    alignment_coarse: torch.Tensor
    edge: torch.Tensor
    edge_enabled: bool

    def detached_scalars(self) -> dict[str, float | bool]:
        return {
            "total": float(self.total.detach().item()),
            "final": float(self.final.detach().item()),
            "alignment_full": float(self.alignment_full.detach().item()),
            "alignment_mid": float(self.alignment_mid.detach().item()),
            "alignment_coarse": float(self.alignment_coarse.detach().item()),
            "edge": float(self.edge.detach().item()),
            "edge_enabled": self.edge_enabled,
        }


class DeTurbLoss(nn.Module):
    def __init__(self, config: DeTurbLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeTurbLossConfig()
        self.charbonnier = CharbonnierLoss(self.config.charbonnier_epsilon)
        self.edge = LaplacianEdgeLoss(self.config.charbonnier_epsilon)

    def forward(
        self,
        output: DeTurbOutput,
        target_reference: torch.Tensor,
        *,
        iteration: int,
    ) -> LossBreakdown:
        if not isinstance(output, DeTurbOutput):
            raise TypeError("DeTurbLoss requires auxiliary DeTurbOutput")
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if output.restored.shape != target_reference.shape or target_reference.ndim not in (4, 5):
            raise ValueError("target must match restored BCHW or BCTHW output")
        # A full-clip model removes distortion in each frame's own coordinates.
        # Broadcasting the center GT here would train all outputs toward one
        # instant and erase object motion.
        alignment_target = (
            target_reference if target_reference.ndim == 5
            else target_reference.unsqueeze(2).expand_as(output.registration.aligned)
        )
        final_loss = self.charbonnier(output.restored, target_reference)
        alignment_full = self.charbonnier(
            output.registration.aligned,
            alignment_target,
        )
        alignment_mid = self.charbonnier(
            output.registration.aligned_mid,
            alignment_target,
        )
        alignment_coarse = self.charbonnier(
            output.registration.aligned_coarse,
            alignment_target,
        )
        stage_weights = self.config.alignment_stage_weights
        alignment_total = (
            stage_weights[0] * alignment_full
            + stage_weights[1] * alignment_mid
            + stage_weights[2] * alignment_coarse
        )
        edge_enabled = (
            self.config.edge_weight > 0
            and iteration >= self.config.edge_start_iteration
        )
        if edge_enabled:
            prediction, target = output.restored, target_reference
            if prediction.ndim == 5:
                b, c, t, h, w = prediction.shape
                prediction = prediction.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                target = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            edge_loss = self.edge(prediction, target)
        else:
            edge_loss = final_loss.new_zeros(())
        total = (
            self.config.final_weight * final_loss
            + self.config.alignment_weight * alignment_total
            + self.config.edge_weight * edge_loss
        )
        return LossBreakdown(
            total=total,
            final=final_loss,
            alignment_full=alignment_full,
            alignment_mid=alignment_mid,
            alignment_coarse=alignment_coarse,
            edge=edge_loss,
            edge_enabled=edge_enabled,
        )
