"""Exact Farneback/cv2.remap protocol from warping-error-standard."""
import math

import numpy as np

from deturb.utils.warping_error_reference import (
    estimate_flow, warp_array, forward_backward_mask, masked_rgb_l1,
)


def tensor_rgb(tensor):
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError('Expected one RGB BCHW image')
    image = tensor.detach()[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Warping inputs must be finite RGB [0,1]')
    return image


class FarnebackGroundTruthMotion:
    """Share the reference flow/mask across models, preserving native pixels."""
    def __init__(self):
        self.cached_identical = None

    def reset(self):
        self.cached_identical = None

    def motion(self, previous, current):
        a, b = tensor_rgb(previous), tensor_rgb(current)
        identical = np.array_equal(a, b)
        if identical and self.cached_identical is not None:
            old, result = self.cached_identical
            if np.array_equal(a, old):
                return result
        forward = estimate_flow(a, b, 'farneback', 0, 'medium')
        backward = estimate_flow(b, a, 'farneback', 0, 'medium')
        warped_gt, bounds = warp_array(b, forward)
        mask = forward_backward_mask(forward, backward, bounds, .01, .5)
        gt_error, pixels = masked_rgb_l1(a, warped_gt, mask)
        if not pixels or not math.isfinite(gt_error):
            raise RuntimeError('Empty/invalid warp mask for adjacent GT pair')
        flow_magnitude = float(np.sqrt(np.sum(forward * forward, axis=2))[mask].mean())
        result = {'flow': forward, 'mask': mask, 'valid_pixels': pixels,
                  'gt_warp_error': gt_error, 'mask_ratio': pixels / mask.size,
                  'flow_magnitude': flow_magnitude}
        self.cached_identical = (a.copy(), result) if identical else None
        return result

    def score(self, previous, current, motion):
        a, b = tensor_rgb(previous), tensor_rgb(current)
        warped, _ = warp_array(b, motion['flow'])
        value, pixels = masked_rgb_l1(a, warped, motion['mask'])
        if pixels != motion['valid_pixels'] or not pixels or not math.isfinite(value):
            raise RuntimeError('Empty/invalid prediction warp score')
        return value, abs(value - motion['gt_warp_error'])
