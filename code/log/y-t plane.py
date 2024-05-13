import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_yt_plane(video_path, column_index=None):
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

    # If column_index is not specified, use the middle of the frame
    if column_index is None:
        column_index = frame_width // 2

    # Create an empty array to store the y-t plane (height x number of frames)
    yt_plane = np.zeros((frame_height, frame_count), dtype=np.uint8)

    # Read each frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract the vertical line at column_index
        vertical_line = gray_frame[:, column_index]

        # Store this line in the yt_plane array
        yt_plane[:, frame_idx] = vertical_line

        frame_idx += 1

    cap.release()
    return yt_plane


# Path to your video file
video_path = "C:\\Users\\Zouzh\\Desktop\\2 Defconv Real\\skip_01_SD_crop3.mp4"

# Extract the y-t plane from the video
yt_plane = extract_yt_plane(video_path)

# Plot the y-t plane
plt.figure(figsize=(10, 5))
plt.imshow(yt_plane, aspect='auto', cmap='gray')
plt.title('Y-T Plane Image')
plt.xlabel('Frame')
plt.ylabel('Pixel Intensity along the Vertical Line')
plt.show()
