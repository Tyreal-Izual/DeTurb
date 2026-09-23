"""Safe tensor-to-image conversion helpers."""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW image tensor without mutating its source storage."""

    image = tensor.detach().squeeze().float().cpu().clamp(0, 1).numpy()
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)
