"""Shared complete-coverage window and video metadata helpers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass(frozen=True)
class TemporalWindow:
    input_start: int
    output_start: int
    output_end: int

def compute_tile_starts(length: int, patch_size: int, overlap: int) -> list[int]:
    if length <= 0 or patch_size <= 0:
        raise ValueError("length and patch_size must be positive")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")
    if length <= patch_size:
        return [0]

    stride = patch_size - overlap
    last_start = length - patch_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts

def compute_temporal_windows(
    total_frames: int,
    window_size: int,
) -> list[TemporalWindow]:
    if total_frames <= 0 or window_size <= 0:
        raise ValueError("total_frames and window_size must be positive")
    if total_frames <= window_size:
        return [TemporalWindow(0, 0, total_frames)]

    stride = max(window_size // 2, 1)
    last_start = total_frames - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    windows = []
    for index, start in enumerate(starts):
        if index == 0:
            output_start = 0
        else:
            previous_start = starts[index - 1]
            output_start = (previous_start + window_size + start) // 2

        if index == len(starts) - 1:
            output_end = total_frames
        else:
            next_start = starts[index + 1]
            output_end = (start + window_size + next_start) // 2

        if not start <= output_start <= output_end <= start + window_size:
            raise RuntimeError("Temporal window boundaries are inconsistent")
        windows.append(TemporalWindow(start, output_start, output_end))
    return windows

def probe_video(input_path: str | Path) -> tuple[int, float, int, int]:
    path = Path(input_path).expanduser()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Failed to open video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        capture.release()
    if frame_count <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frame_count, fps, height, width
