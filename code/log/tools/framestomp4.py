import cv2
import os

def images_to_video(input_folder, output_video_path, fps=30):
    # Retrieve the list of image files
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()  # Sort the images by name

    # Read the first image to obtain the dimensions
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(input_folder, image))
        out.write(frame)  # Write the frame to the video

    # Release everything when job is finished
    out.release()
    print(f'Video saved as {output_video_path}')

# Usage
images_to_video("C:\\Users\\Zouzh\\Desktop\\CLEAR\\VayTekTiffs_restored", "C:\\Users\\Zouzh\\Desktop\\Frederick_Zou Individual Project\\image\\CLEAR\\input\\VayTekTiffs.mp4", fps=24)
