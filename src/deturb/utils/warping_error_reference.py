"""GT-flow Farneback math from the user-defined warping-error protocol."""
from __future__ import annotations
import cv2
import numpy as np

def rgb01_to_gray_u8(frame: np.ndarray) -> np.ndarray:
    rgb = np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def resize_for_flow(frame: np.ndarray, long_edge: int) -> tuple[np.ndarray, float, float]:
    if long_edge <= 0 or max(frame.shape[:2]) <= long_edge:
        return frame, 1.0, 1.0
    height, width = frame.shape[:2]
    scale = float(long_edge) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, width / new_width, height / new_height

def estimate_flow(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    method: str,
    long_edge: int,
    dis_preset: str,
) -> np.ndarray:
    """Return GT flow from frame_a to frame_b in full-resolution pixel units."""
    height, width = frame_a.shape[:2]
    work_a, scale_x, scale_y = resize_for_flow(frame_a, long_edge)
    work_b, _, _ = resize_for_flow(frame_b, long_edge)
    gray_a = rgb01_to_gray_u8(work_a)
    gray_b = rgb01_to_gray_u8(work_b)
    if method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None, pyr_scale=0.5, levels=5, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
    elif method == "dis":
        if not hasattr(cv2, "DISOpticalFlow_create"):
            raise RuntimeError("OpenCV DIS optical flow is unavailable")
        preset = getattr(cv2, f"DISOPTICAL_FLOW_PRESET_{dis_preset.upper()}")
        flow = cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)
    elif method == "tvl1":
        if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
            estimator = cv2.optflow.DualTVL1OpticalFlow_create()
        elif hasattr(cv2, "DualTVL1OpticalFlow_create"):
            estimator = cv2.DualTVL1OpticalFlow_create()
        else:
            raise RuntimeError("OpenCV TV-L1 optical flow is unavailable")
        flow = estimator.calc(gray_a, gray_b, None)
    else:
        raise ValueError(method)
    if flow.shape[:2] != (height, width):
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
    return flow.astype(np.float32, copy=False)

def warp_array(array: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(
        array.astype(np.float32, copy=False), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return warped, valid

def forward_backward_mask(
    flow_fw: np.ndarray,
    flow_bw: np.ndarray,
    bounds_mask: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    warped_bw, bw_valid = warp_array(flow_bw, flow_fw)
    residual = flow_fw + warped_bw
    residual_sq = np.sum(residual * residual, axis=2)
    magnitude_sq = np.sum(flow_fw * flow_fw, axis=2) + np.sum(warped_bw * warped_bw, axis=2)
    return bounds_mask & bw_valid & (residual_sq <= alpha * magnitude_sq + beta)

def masked_rgb_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    pixels = int(mask.sum())
    if pixels == 0:
        return float("nan"), 0
    per_pixel = np.abs(a - b).mean(axis=2)
    return float(per_pixel[mask].mean()), pixels
