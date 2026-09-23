"""Static image sequences using the existing full-clip augmentation contract."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from deturb.data.dataset_video_train import DataLoaderTurbVideo


class DataLoaderTurbStatic(DataLoaderTurbVideo):
    def __init__(self, root_dir, num_frames=12, patch_size=None, noise=None,
                 is_train=True, manifest_path=None, manifest_split=None,
                 read_attempts=3, video_timeout_ms=10000, retry_delay_seconds=2.0,
                 data_layout='static_frames'):
        if data_layout != 'static_frames' or not manifest_path or not manifest_split:
            raise ValueError('Static clips require a static_frames manifest and split')
        if num_frames <= 0 or read_attempts < 1 or retry_delay_seconds < 0:
            raise ValueError('Invalid static clip/retry settings')
        root = Path(root_dir).expanduser().resolve()
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get('data_layout') != 'static_frames':
            raise ValueError('Static manifest layout mismatch')
        if Path(manifest['split_roots'][manifest_split]).resolve() != root:
            raise ValueError('Static manifest split root mismatch')
        samples = manifest['splits'][manifest_split]['samples']
        names = [s['name'] for s in samples]
        if not names or len(names) != len(set(names)):
            raise ValueError('Empty or duplicate static samples')
        self.gt_list, self.turb_list, self.frame_paths = [], [], []
        for sample in samples:
            name, frames = sample['name'], sample['frame_names']
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('Invalid static sample name')
            if len(frames) < num_frames or len(frames) != len(set(frames)):
                raise ValueError('Short or duplicate static frame list')
            if any(Path(f).name != f or f in ('.', '..') for f in frames):
                raise ValueError('Invalid static frame filename')
            gt = root / name / 'gt.jpg'
            paths = [root / name / 'turb' / f for f in frames]
            if not gt.is_file() or not all(p.is_file() for p in paths):
                raise FileNotFoundError(f'Incomplete static sample: {name}')
            self.gt_list.append(str(gt))
            self.turb_list.append(str(root / name / 'turb'))
            self.frame_paths.append(paths)
        self.blur_list = []
        self.data_layout = data_layout
        self.num_frames, self.ps, self.noise = num_frames, patch_size, noise
        self.train, self.sizex = is_train, len(samples)
        self.read_attempts = read_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.video_timeout_ms = video_timeout_ms

    def fingerprint_paths(self, index):
        return [Path(self.gt_list[index]), *self.frame_paths[index]]

    def _read_aligned_clip_once(self, idx, random_start):
        paths = self.frame_paths[idx]
        start = (random.randint(0, len(paths) - self.num_frames) if random_start
                 else (len(paths) - self.num_frames) // 2)
        def read(path):
            with Image.open(path) as im:
                return np.asarray(im.convert('RGB'))[:, :, ::-1].copy()
        gt = read(self.gt_list[idx])
        frames = [read(p) for p in paths[start:start + self.num_frames]]
        if any(f.shape != gt.shape for f in frames):
            raise ValueError(f'Static input/GT geometry mismatch: {self.gt_list[idx]}')
        # Slot zero is unused by task=turb; retain the Dynamic augmentation RNG
        # order. Each actual output is supervised by the same clean static GT.
        return frames, frames, [gt] * self.num_frames, gt.shape[0], gt.shape[1]
