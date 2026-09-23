"""Checked sequence decoding and file identities."""
from pathlib import Path
import hashlib
import time
import cv2
import numpy as np
import torch

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def decode_cache(path, destination, metadata, count, attempts=3):
    """Retry the same complete video before inference; use lossless local storage."""
    for attempt in range(attempts):
        cap = cv2.VideoCapture()
        array = None
        try:
            if not cap.open(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                                                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000]):
                raise OSError(f'Cannot open {path}')
            actual = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            expected = tuple(metadata[k] for k in ('frames', 'height', 'width'))
            if actual != expected:
                raise ValueError(f'Manifest/video mismatch: {path}: {actual} != {expected}')
            array = np.lib.format.open_memmap(destination, mode='w+', dtype=np.uint8,
                                             shape=(count, *actual[1:], 3))
            for i in range(count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise OSError(f'Cannot decode {path}, frame {i}')
                array[i] = frame
            array.flush()
            return array
        except (OSError, cv2.error) as error:
            if attempt + 1 == attempts:
                raise
            print(f'Decode retry {attempt+1}/{attempts}: {error}', flush=True)
            time.sleep(2 * (attempt + 1))
        finally:
            cap.release()

def input_rgb(frame, device):
    return torch.from_numpy(np.ascontiguousarray(frame[..., ::-1])).permute(2, 0, 1).to(device).float() / 255
