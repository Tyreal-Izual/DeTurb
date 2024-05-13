import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_xt_plane(video_path, line_index=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Determine the number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the height and width of the video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # If line_index is not specified, use the middle of the frame
    if line_index is None:
        line_index = frame_height // 2

    # Create an empty array to store the x-t plane (width x number of frames)
    xt_plane = np.zeros((frame_count, frame_width), dtype=np.uint8)

    # Read each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract the horizontal line at line_index
        horizontal_line = gray_frame[line_index, :]

        # Store this line in the xt_plane array
        xt_plane[frame_idx, :] = horizontal_line

        frame_idx += 1

    cap.release()
    return xt_plane


# Path to your video file
video_path = "C:\\Users\\Zouzh\\Desktop\\2 Defconv Real\\cueva_01_SD_crop1.mp4"

# Extract the x-t plane from the video
xt_plane = extract_xt_plane(video_path)

# Plot the x-t plane
plt.figure(figsize=(5, 10))
plt.imshow(xt_plane, aspect='auto', cmap='gray')
plt.title('X-T Plane Image')
plt.xlabel('Frame')
plt.ylabel('Pixel Intensity along the Horizontal Line')
plt.show()
