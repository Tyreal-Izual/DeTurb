"""Streaming perceptual metrics with continuous, zero-padded frame identities."""
import logging
import math

from deturb.utils.perceptual_video import lpips_distance

LOGGER = logging.getLogger(__name__)


class PerceptualAccumulator:
    """Frame index i corresponds to lexical name f'{i:08d}.png'.

    Inputs are RGB BCHW floating tensors in [0,1], without uint8 rounding or
    spatial resizing. Missing GT mappings skip adjacent pairs without bridging.
    This implements the same pair semantics as compute_tlpips, without writing
    intermediate restored frames or applying lossy image/video encoding.
    """
    def __init__(self, model, video, debug=False):
        self.model, self.video, self.debug = model, video, debug
        self.frames = self.lpips_frames = self.pairs = 0
        self.lpips_sum = self.tlpips_sum = 0.
        self.previous = None

    def update(self, index, pred, gt, gt_adjacent_distance=None):
        if self.debug and index >= 10:
            return
        if index != self.frames:
            raise ValueError('Noncontiguous output index would bridge or repeat frames')
        name = f'{index:08d}.png'
        if gt is not None:
            self.lpips_sum += lpips_distance(self.model, pred, gt)
            self.lpips_frames += 1
        if self.previous is not None:
            old_name, old_pred, old_gt = self.previous
            if old_gt is None or gt is None:
                LOGGER.warning('GT not found for %s / %s in %s, skipping pair', old_name, name, self.video)
            else:
                reference = (lpips_distance(self.model, old_gt, gt) if gt_adjacent_distance is None
                             else gt_adjacent_distance)
                distance = lpips_distance(self.model, old_pred, pred)
                value = abs(distance - reference)
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite temporal LPIPS score')
                self.tlpips_sum += value
                self.pairs += 1
                if self.debug:LOGGER.info('tLPIPS score: %s', value)
        self.previous = (name, pred, gt)
        self.frames += 1

    def finish(self):
        lpips = self.lpips_sum/self.lpips_frames if self.lpips_frames else math.nan
        tlpips = self.tlpips_sum/self.pairs if self.pairs else math.nan
        LOGGER.info('tLPIPS video: %s, find %d frames; valid pairs=%d; tLPIPS score: %s',
                    self.video, self.frames, self.pairs, tlpips)
        return {'lpips': lpips, 'tlpips': tlpips, 'frames': self.frames,
                'lpips_frames': self.lpips_frames, 'valid_pairs': self.pairs,
                'lpips_sum': self.lpips_sum, 'tlpips_sum': self.tlpips_sum,
                'temporal_status': 'valid' if self.pairs else 'invalid_no_valid_pairs'}
