"""User-standard tLPIPS, with ordinary frame LPIPS using the same model."""
from __future__ import annotations
import logging
import math
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def lpips_distance(model, a, b):
    """RGB BCHW [0,1] -> [-1,1] exactly once, without resizing."""
    model.eval()
    value = float(model(a * 2 - 1, b * 2 - 1).item())
    if hasattr(model, 'reset'):
        model.reset()
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite LPIPS distance')
    return value


def temporal_pair(model, p0, p1, g0, g1):
    return abs(lpips_distance(model, p0, p1) - lpips_distance(model, g0, g1))


# ---------- tLPIPS ----------
def compute_tlpips(video_dir, name_dict, *, lpips_model, device='cpu', debug=False):
    frames = sorted(os.listdir(video_dir))
    if any(Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
           or not (Path(video_dir)/name).is_file() for name in frames):
        raise ValueError('tLPIPS directory must contain only video frames')
    if debug:
        frames = frames[:10]
    LOGGER.info('tLPIPS video: %s, find %d frames.', video_dir, len(frames))
    scores = []
    def read(path):
        with Image.open(path) as image:
            return to_tensor(image.convert('RGB')).unsqueeze(0).to(device)
    for a, b in zip(frames, frames[1:]):
        ga, gb = name_dict.get(a), name_dict.get(b)
        if ga is None or gb is None:
            LOGGER.warning('GT not found for %s, %s, skipping...', ga, gb)
            continue
        value = temporal_pair(lpips_model, read(Path(video_dir)/a), read(Path(video_dir)/b), read(ga), read(gb))
        scores.append(value)
        if debug: LOGGER.info('tLPIPS score: %s', value)
    result = sum(scores)/len(scores) if scores else math.nan
    LOGGER.info('tLPIPS score: %s (valid pairs=%d)', result, len(scores))
    return result
