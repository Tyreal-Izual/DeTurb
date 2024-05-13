import cv2
import numpy as np

def draw_zoomed_area_with_highlighted_lines(image_path, output_path, start_x, start_y, width, height):
    """
    Draws an image with a zoomed area indicated by lines connecting it to the original region, and highlights the zoomed area in the original image.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path where the modified image will be saved.
        start_x (int): The x-coordinate of the upper-left corner of the zoom area.
        start_y (int): The y-coordinate of the upper-left corner of the zoom area.
        width (int): The width of the rectangle to zoom into.
        height (int): The height of the rectangle to zoom into.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # Highlight the zoom area in the original image
    cv2.rectangle(image, (start_x, start_y), (start_x + width, start_y + height), (0, 255, 0), 2)

    # Define the region of interest and zoom in
    roi = image[start_y:start_y + height, start_x:start_x + width]
    zoomed_image = cv2.resize(roi, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

    # Create an image that combines the original and the zoomed image
    combined_height = image.shape[0] + zoomed_image.shape[0] + 10  # add some space
    combined_image = np.zeros((combined_height, image.shape[1], 3), dtype=np.uint8)
    combined_image[:image.shape[0], :image.shape[1]] = image

    # Position for the zoomed image
    start_y_zoom = image.shape[0] + 10
    combined_image[start_y_zoom:start_y_zoom + zoomed_image.shape[0], :zoomed_image.shape[1]] = zoomed_image

    # Drawing lines
    # Top left to bottom left (line start point to zoom box top left)
    cv2.line(combined_image, (start_x, start_y), (0, start_y_zoom), (255, 0, 0), 2)
    # Top right to bottom right (line end point to zoom box top right)
    cv2.line(combined_image, (start_x + width, start_y), (width*2, start_y_zoom), (255, 0, 0), 2)

    # Save the combined image
    cv2.imwrite(output_path, combined_image)
    print("Zoomed and annotated image saved to", output_path)




draw_zoomed_area_with_highlighted_lines("C:\\Users\\Zouzh\\Desktop\\Frederick_Zou Individual Project\\image\\real-world example\\dynamic\\RoadMirage_SD_Clip02-2-stage-0166.png", "C:\\Users\\Zouzh\\Desktop\\Frederick_Zou Individual Project\\image\\real-world example\\zoomin\\RoadMirage_SD_Clip02-2-stage-0166.png", 10, 262, 345, 123)

