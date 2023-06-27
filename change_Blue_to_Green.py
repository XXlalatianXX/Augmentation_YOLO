import cv2
import os
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

input_dir = "D:/Frompop2(Blue)"
output_dir = "D:/Frompop2(Green)"
os.makedirs(output_dir, exist_ok=True)

image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))

def change_blue_to_green(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue color range in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue pixels
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Modify the hue channel of blue pixels to green
    hsv_image[:, :, 0][blue_mask != 0] = 60  # Green hue value in HSV

    # Convert the modified image back to BGR color space
    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return modified_image

def process_image(image_path):
    filename = os.path.basename(image_path)
    annotation_path = os.path.join(input_dir, os.path.splitext(filename)[0] + ".txt")

    if os.path.isfile(annotation_path):
        with open(annotation_path, "r") as annotation_file:
            annotation_data = annotation_file.read()

        image = cv2.imread(image_path)

        modified_image = change_blue_to_green(image)

        new_filename = os.path.splitext(filename)[0] + "_modified"
        new_image_path = os.path.join(output_dir, new_filename + ".jpg")
        new_annotation_path = os.path.join(output_dir, new_filename + ".txt")

        cv2.imwrite(new_image_path, modified_image)

        with open(new_annotation_path, "w") as new_annotation_file:
            new_annotation_file.write(annotation_data)

        print(f"Modified image saved: {new_image_path}")
        print(f"New annotation file saved: {new_annotation_path}")
    else:
        print(f"Annotation file not found for image: {image_path}")

# Set the number of threads for concurrent execution
num_threads = 4

# Create a ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Process the images using concurrent execution
    executor.map(process_image, image_files)

exit()
