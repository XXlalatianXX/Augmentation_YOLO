import cv2
import os
import glob
import random
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import imgaug.augmenters as iaa


input_dir = "D:/Pom_All"
output_dir = "E:/Darknet_Using/darknet/build/darknet/x64/data/obj"

os.makedirs(output_dir, exist_ok=True)

# Define the color manipulation augmentations
color_augmentations = iaa.Sequential([
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
])

image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))

def process_image(image_path):
    filename = os.path.basename(image_path)
    annotation_path = os.path.join(input_dir, os.path.splitext(filename)[0] + ".txt")

    if os.path.isfile(annotation_path):
        with open(annotation_path, "r") as annotation_file:
            annotation_data = annotation_file.read()

        image = cv2.imread(image_path)

        # Apply color manipulation augmentations
        augmented_image = color_augmentations(image=image)

        new_filename = os.path.splitext(filename)[0] + "_augcolorMa"
        new_image_path = os.path.join(output_dir, new_filename + ".jpg")
        new_annotation_path = os.path.join(output_dir, new_filename + ".txt")

        cv2.imwrite(new_image_path, augmented_image)

        with open(new_annotation_path, "w") as new_annotation_file:
            new_annotation_file.write(annotation_data)

        print(f"Augmented image saved: {new_image_path}")
        print(f"New annotation file saved: {new_annotation_path}")

# Set the number of threads for concurrent execution
num_threads = 4

# Create a ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Process the images using concurrent execution
    executor.map(process_image, image_files)

exit()