import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch


def load_video_frames(video_path):
    """Load video frames into a list of numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def calculate_metrics(frames1, frames2):
    """Calculate SSIM, PSNR, and LPIPS for two lists of frames."""
    ssim_values = []
    psnr_values = []
    lpips_model = lpips.LPIPS(net='alex')  # Using AlexNet
    lpips_values = []

    for frame1, frame2 in zip(frames1, frames2):
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        ssim_val = ssim(frame1_gray, frame2_gray, data_range=frame2_gray.max() - frame2_gray.min())
        psnr_val = psnr(frame1_gray, frame2_gray, data_range=frame2_gray.max() - frame2_gray.min())

        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().div(255)
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().div(255)

        lpips_val = lpips_model(frame1_tensor, frame2_tensor)

        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)
        lpips_values.append(lpips_val.item())

    return np.mean(ssim_values), np.mean(psnr_values), np.mean(lpips_values)


# Load frames from videos
frames_video1 = load_video_frames("C:\\Users\\Zouzh\\Desktop\\Frederick_Zou Individual Project\\image\\CLEAR\\a\\gt\\Airport.mp4")
frames_video2 = load_video_frames("C:\\Users\\Zouzh\\Desktop\\Frederick_Zou Individual Project\\image\\CLEAR\\tmt\\Airport.mp4")

# Calculate metrics
avg_ssim, avg_psnr, avg_lpips = calculate_metrics(frames_video1, frames_video2)
print(f"Average SSIM: {avg_ssim}")
print(f"Average PSNR: {avg_psnr}")
print(f"Average LPIPS: {avg_lpips}")
