import cv2
import numpy as np
import skvideo.io
import skvideo.measure
import os
from skimage.transform import resize

def load_video(video_path):
    """ Load video using skvideo. """
    videodata = skvideo.io.vread(video_path)
    return videodata

def calculate_niqe_scores(video_frames):
    """ Calculate NIQE scores using a manual resizing step to avoid deprecated function calls. """
    scores = []
    for frame in video_frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_frame = resize(gray_frame, (gray_frame.shape[0] // 2, gray_frame.shape[1] // 2),
                               anti_aliasing=True, preserve_range=True)
        input_frame = resized_frame[np.newaxis, :, :, np.newaxis]
        score = skvideo.measure.niqe(input_frame)
        scores.append(score)
    return scores

def list_videos(directory):
    """ List all video files in a directory. """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov'))]

def main(folder1, folder2, folder3):
    # List videos in each folder
    videos_folder1 = list_videos(folder1)
    videos_folder2 = list_videos(folder2)
    videos_folder3 = list_videos(folder3)

    # Check if all folders have the same number of video files
    if len(videos_folder1) != len(videos_folder2) or len(videos_folder2) != len(videos_folder3):
        print("Error: The folders do not contain the same number of videos.")
        return

    # Iterate over paired videos
    for video_path1, video_path2, video_path3 in zip(videos_folder1, videos_folder2, videos_folder3):
        # Load videos
        video1 = load_video(video_path1)
        video2 = load_video(video_path2)
        video3 = load_video(video_path3)

        # Calculate NIQE scores for each video
        scores_video1 = calculate_niqe_scores(video1)
        scores_video2 = calculate_niqe_scores(video2)
        scores_video3 = calculate_niqe_scores(video3)

        # Compute average NIQE score for each video
        avg_score1 = np.mean(scores_video1)
        avg_score2 = np.mean(scores_video2)
        avg_score3 = np.mean(scores_video3)

        print(f"Average NIQE Score for {os.path.basename(video_path1)}: {avg_score1:.2f}")
        print(f"Average NIQE Score for {os.path.basename(video_path2)}: {avg_score2:.2f}")
        print(f"Average NIQE Score for {os.path.basename(video_path3)}: {avg_score3:.2f}")

if __name__ == '__main__':
    # Specify the paths to the three folders
    folder1 = 'C:\\Users\\Zouzh\\Desktop\\New folder (5)'
    folder2 = 'C:\\Users\\Zouzh\\Desktop\\New folder (5)'
    folder3 = 'C:\\Users\\Zouzh\\Desktop\\New folder (4)'  # Add path for the third folder here
    main(folder1, folder2, folder3)
