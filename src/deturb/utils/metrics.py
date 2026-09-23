"""Batched tensor metrics matching the legacy uint8 PSNR/SSIM semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .numerics import require_finite


PSNR_MAX_DB = 100.0
METRIC_POLICY = {
    "pixel_domain": "RGB uint8 rounded",
    "psnr_aggregation": "mean of frame PSNR capped at 100 dB",
    "psnr_max_db": PSNR_MAX_DB,
    "perfect_frames": "count of frames with zero uint8 MSE",
}


@dataclass(frozen=True)
class MetricTotals:
    psnr_sum: float
    ssim_sum: float
    count: int
    perfect_frames: int = 0


def _gaussian_kernel(
    device: torch.device,
    dtype: torch.dtype,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    coordinates = torch.arange(
        kernel_size,
        device=device,
        dtype=dtype,
    ) - kernel_size // 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)


def batch_psnr_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> MetricTotals:
    """Aggregate frame metrics for [B, T, C, H, W] tensors on-device."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("Expected matching [B, T, C, H, W] tensors")
    if min(prediction.shape) <= 0:
        raise ValueError("Metric tensor dimensions must be positive")
    require_finite((prediction, target), "image metrics", prediction.device)
    if prediction.shape[-2] < 11 or prediction.shape[-1] < 11:
        raise ValueError("SSIM requires frame dimensions of at least 11x11")

    batch, frames, channels, height, width = prediction.shape
    prediction_255 = prediction.detach().clamp(0, 1).mul(255).round()
    target_255 = target.detach().clamp(0, 1).mul(255).round()
    prediction_255 = prediction_255.reshape(
        batch * frames,
        channels,
        height,
        width,
    ).float()
    target_255 = target_255.reshape(
        batch * frames,
        channels,
        height,
        width,
    ).float()

    mse = (prediction_255 - target_255).square().mean(dim=(1, 2, 3))
    mse_floor = 255.0 ** 2 * 10.0 ** (-PSNR_MAX_DB / 10.0)
    psnr = (20 * torch.log10(255.0 / torch.sqrt(mse.clamp_min(mse_floor)))).clamp_max(PSNR_MAX_DB)

    kernel = _gaussian_kernel(prediction.device, prediction_255.dtype)
    window = kernel.expand(channels, 1, 11, 11)

    def filter_image(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, window, groups=channels)

    mu_prediction = filter_image(prediction_255)
    mu_target = filter_image(target_255)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    sigma_prediction = filter_image(prediction_255.square()) - mu_prediction_sq
    sigma_target = filter_image(target_255.square()) - mu_target_sq
    sigma_product = filter_image(prediction_255 * target_255) - mu_product
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim_map = (
        (2 * mu_product + c1) * (2 * sigma_product + c2)
        / (
            (mu_prediction_sq + mu_target_sq + c1)
            * (sigma_prediction + sigma_target + c2)
        )
    )
    ssim = ssim_map.mean(dim=(1, 2, 3))
    return MetricTotals(
        psnr_sum=float(psnr.sum().item()),
        ssim_sum=float(ssim.sum().item()),
        count=batch * frames,
        perfect_frames=int((mse == 0).sum().item()),
    )


def batch_image_psnr_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> MetricTotals:
    """Aggregate image metrics for matching `[B,C,H,W]` tensors."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Expected matching [B,C,H,W] tensors")
    return batch_psnr_ssim(
        prediction.unsqueeze(1),
        target.unsqueeze(1),
    )


def batch_model_output_psnr_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> MetricTotals:
    """Score BCHW reference images or every frame of BCTHW model outputs."""
    if prediction.ndim == 5 and target.ndim == 5:
        return batch_psnr_ssim(
            prediction.permute(0, 2, 1, 3, 4),
            target.permute(0, 2, 1, 3, 4),
        )
    return batch_image_psnr_ssim(prediction, target)
