"""Static RGB image decoding with exact geometry checks."""
import numpy as np
from PIL import Image

def image_bgr(path, shape):
    with Image.open(path) as image:
        frame = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
    if frame.shape != (*shape, 3):
        raise ValueError(f'Image/manifest geometry mismatch: {path}')
    return frame
