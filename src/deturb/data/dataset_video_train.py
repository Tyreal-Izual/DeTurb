from logging import root
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import cv2
import logging
import time
from pathlib import Path

from deturb.utils.manifest import load_manifest_names


class DataLoaderTurbVideo(Dataset):
    VIDEO_EXTENSIONS = {'.avi', '.mkv', '.mov', '.mp4'}

    def __init__(self, root_dir, num_frames=12, patch_size=None, noise=None,
                 is_train=True, manifest_path=None, manifest_split=None,
                 read_attempts=3, video_timeout_ms=10000, retry_delay_seconds=2.0,
                 data_layout='triplet'):
        super(DataLoaderTurbVideo, self).__init__()
        if data_layout not in ('triplet', 'paired_turb'):
            raise ValueError('data_layout must be triplet or paired_turb')
        self.data_layout = data_layout
        if read_attempts < 1 or video_timeout_ms <= 0 or retry_delay_seconds < 0:
            raise ValueError('Invalid video read retry/timeout settings')
        self.read_attempts = read_attempts
        self.video_timeout_ms = video_timeout_ms
        self.retry_delay_seconds = retry_delay_seconds
        self.num_frames = num_frames
        root_path = Path(root_dir).expanduser()
        kinds = ('gt', 'turb', 'blur') if data_layout == 'triplet' else ('gt', 'turb')
        data_dirs = {kind: root_path / kind for kind in kinds}
        missing_dirs = [str(path) for path in data_dirs.values() if not path.is_dir()]
        if missing_dirs:
            raise FileNotFoundError(
                'Missing required dynamic dataset directories: '
                + ', '.join(missing_dirs)
            )

        if manifest_path:
            if not manifest_split:
                raise ValueError('manifest_split is required with manifest_path')
            video_names = load_manifest_names(manifest_path, manifest_split)
        else:
            video_names = sorted(
                path.name
                for path in data_dirs['gt'].iterdir()
                if path.is_file() and path.suffix.lower() in self.VIDEO_EXTENSIONS
            )
        if not video_names:
            raise ValueError(f'No supported videos found in {data_dirs["gt"]}')

        missing_pairs = [
            str(data_dirs[kind] / name)
            for name in video_names
            for kind in kinds
            if not (data_dirs[kind] / name).is_file()
        ]
        if missing_pairs:
            preview = ', '.join(missing_pairs[:5])
            suffix = '' if len(missing_pairs) <= 5 else f' (+{len(missing_pairs) - 5} more)'
            raise FileNotFoundError(f'Missing paired dynamic videos: {preview}{suffix}')

        self.gt_list = [str(data_dirs['gt'] / name) for name in video_names]
        self.turb_list = [str(data_dirs['turb'] / name) for name in video_names]
        self.blur_list = ([str(data_dirs['blur'] / name) for name in video_names]
                          if data_layout == 'triplet' else [])

        self.ps = patch_size
        self.sizex = len(self.gt_list)  # get the size of target
        self.train = is_train
        self.noise = noise

    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise):
        noise = (noise**0.5)*torch.randn(img.shape)
        out = img + noise
        return out.clamp(0,1)

    def _read_aligned_clip(self, idx, random_start):
        # A failed read must not choose a different sample/window or consume
        # extra augmentation RNG. Reopen the entire paired triplet each time.
        rng_state = random.getstate()
        for attempt in range(1, self.read_attempts + 1):
            try:
                return self._read_aligned_clip_once(idx, random_start)
            except (OSError, cv2.error) as error:
                random.setstate(rng_state)
                if attempt == self.read_attempts:
                    raise OSError(
                        f'Video read failed after {attempt} attempts for '
                        f'{self.gt_list[idx]}: {error}'
                    ) from error
                delay = self.retry_delay_seconds * attempt
                logging.warning(
                    'Video read attempt %d/%d failed for %s; '
                    'retrying the same triplet in %.1fs: %s',
                    attempt, self.read_attempts, self.gt_list[idx], delay, error,
                )
                time.sleep(delay)

    def _read_aligned_clip_once(self, idx, random_start):
        paths = ([self.blur_list[idx]] if self.data_layout == 'triplet' else []) + [self.turb_list[idx], self.gt_list[idx]]
        captures = []

        try:
            for path in paths:
                capture = cv2.VideoCapture()
                captures.append(capture)
                capture.open(path, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.video_timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.video_timeout_ms,
                ])
                if not capture.isOpened():
                    raise OSError(f'Failed to open video: {path}')

            frame_counts = [
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                for capture in captures
            ]
            if len(set(frame_counts)) != 1:
                raise ValueError(
                    f'Paired videos have different frame counts for {paths}: '
                    f'{frame_counts}'
                )

            total_frames = frame_counts[0]
            if total_frames < self.num_frames:
                raise ValueError(
                    f'Video {self.gt_list[idx]} has {total_frames} frames; '
                    f'{self.num_frames} are required'
                )

            if random_start:
                start_frame_id = random.randint(0, total_frames - self.num_frames)
            else:
                start_frame_id = (total_frames - self.num_frames) // 2

            for capture in captures:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id):
                    raise OSError(f'Failed to seek paired videos: {paths}')

            sequences = []
            for path, capture in zip(paths, captures):
                frames = []
                for frame_offset in range(self.num_frames):
                    success, frame = capture.read()
                    if not success or frame is None:
                        frame_id = start_frame_id + frame_offset
                        raise OSError(
                            f'Failed to decode frame {frame_id} from {path}'
                        )
                    frames.append(frame)
                sequences.append(frames)

            height = int(captures[-1].get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(captures[-1].get(cv2.CAP_PROP_FRAME_WIDTH))
        finally:
            for capture in captures:
                capture.release()

        if self.data_layout == 'paired_turb':
            turb_frames, target_frames = sequences
            # Slot zero is unused by paired-turb training. Keep its augmentation
            # draws so the actual turb/GT tensors match the triplet RNG policy.
            blur_frames = turb_frames
        else:
            blur_frames, turb_frames, target_frames = sequences
        return blur_frames, turb_frames, target_frames, height, width

    def _fetch_chunk_val(self, idx):
        ps = self.ps
        blur_imgs, turb_imgs, tar_imgs, h, w = self._read_aligned_clip(
            idx,
            random_start=False,
        )

        tar_imgs =  [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs]
        turb_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs]
        blur_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in blur_imgs]

        if ps > 0:
            padw = ps-w if w<ps else 0
            padh = ps-h if h<ps else 0
            if padw!=0 or padh!=0:
                blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
                turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]
                tar_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in tar_imgs]

            blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]

            hh, ww = tar_imgs[0].shape[1], tar_imgs[0].shape[2]

            rr     = (hh-ps) // 2
            cc     = (ww-ps) // 2
            # Crop patch
            blur_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in blur_imgs]
            turb_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in turb_imgs]
            tar_imgs  = [img[:, rr:rr+ps, cc:cc+ps] for img in tar_imgs]
        else:
            blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return blur_imgs, turb_imgs, tar_imgs

    def _fetch_chunk_train(self, idx):
        ps = self.ps
        blur_imgs, turb_imgs, tar_imgs, h, w = self._read_aligned_clip(
            idx,
            random_start=True,
        )
        tar_imgs =  [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs]
        turb_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs]
        blur_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in blur_imgs]
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0
        if padw!=0 or padh!=0:
            blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]
            tar_imgs  = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in tar_imgs]

        aug    = random.randint(0, 2)
        if aug == 1:
            blur_imgs = [TF.adjust_gamma(img, 1) for img in blur_imgs]
            turb_imgs = [TF.adjust_gamma(img, 1) for img in turb_imgs]
            tar_imgs  = [TF.adjust_gamma(img, 1) for img in tar_imgs]

        aug    = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_imgs = [TF.adjust_saturation(img, sat_factor) for img in blur_imgs]
            turb_imgs = [TF.adjust_saturation(img, sat_factor) for img in turb_imgs]
            tar_imgs  = [TF.adjust_saturation(img, sat_factor) for img in tar_imgs]

        hh, ww = h, w

        enlarge_factor = random.choice([0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1,2, 1.5, 1.8, 2])
        crop_size = ps * enlarge_factor
        crop_size = min(hh, ww, crop_size)
        hcro = int(crop_size * random.uniform(1.1, 0.9))
        wcro = int(crop_size * random.uniform(1.1, 0.9))
        hcro = min(hcro, hh)
        wcro = min(wcro, ww)
        rr   = random.randint(0, hh-hcro)
        cc   = random.randint(0, ww-wcro)

        # Crop patch
        blur_imgs = [TF.resize(img.crop((cc, rr, cc+wcro, rr+hcro)), (ps, ps)) for img in blur_imgs]
        turb_imgs = [TF.resize(img.crop((cc, rr, cc+wcro, rr+hcro)), (ps, ps)) for img in turb_imgs]
        tar_imgs  = [TF.resize(img.crop((cc, rr, cc+wcro, rr+hcro)), (ps, ps)) for img in tar_imgs]

        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]

        aug    = random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            blur_imgs = [img.flip(1) for img in blur_imgs]
            turb_imgs = [img.flip(1) for img in turb_imgs]
            tar_imgs  = [img.flip(1) for img in tar_imgs]
        elif aug==2:
            blur_imgs = [img.flip(2) for img in blur_imgs]
            turb_imgs = [img.flip(2) for img in turb_imgs]
            tar_imgs  = [img.flip(2) for img in tar_imgs]
        elif aug==3:
            blur_imgs = [torch.rot90(img, dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img, dims=(1,2)) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img, dims=(1,2)) for img in tar_imgs]
        elif aug==4:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img,dims=(1,2), k=2) for img in tar_imgs]
        elif aug==5:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img,dims=(1,2), k=3) for img in tar_imgs]
        elif aug==6:
            blur_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img.flip(1), dims=(1,2)) for img in tar_imgs]
        elif aug==7:
            blur_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img.flip(2), dims=(1,2)) for img in tar_imgs]
        return blur_imgs, turb_imgs, tar_imgs

    def __getitem__(self, index):
        index_ = index % self.sizex
        if self.train:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_train(index_)
        else:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_val(index_)
        return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0)
