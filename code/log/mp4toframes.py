import cv2
import os


def video_to_images(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        # Read frame by frame
        success, frame = cap.read()

        # If the frame is read correctly, save it as an image
        if success:
            # Define the filename for each image
            filename = f'1-{frame_count:04d}.png'
            filepath = os.path.join(output_folder, filename)

            # Save the image
            cv2.imwrite(filepath, frame)
            frame_count += 1
        else:
            # If no frame is read (video end), break the loop
            break

    # Release the video capture object
    cap.release()
    print(f'Images are saved in {output_folder}. Total frames: {frame_count}')


# Usage
video_to_images("C:\\Users\\Zouzh\\Desktop\\Frederick_Zou Individual Project\image\\real-world example\\realworldexample\\1.mp4", "C:\\Users\\Zouzh\\Desktop\\New folder (2)")
