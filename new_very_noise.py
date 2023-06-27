import cv2
import os
import glob
import random
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor


input_dir = "D:/Pom_All"
output_dir = "E:/Darknet_Using/darknet/build/darknet/x64/data/obj"
os.makedirs(output_dir, exist_ok=True)

noise_range = (100, 200)

image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))

def process_image(image_path):
    filename = os.path.basename(image_path)
    annotation_path = os.path.join(input_dir, os.path.splitext(filename)[0] + ".txt")

    if os.path.isfile(annotation_path):
        with open(annotation_path, "r") as annotation_file:
            annotation_data = annotation_file.read()

        image = cv2.imread(image_path)

        # Random noise augmentation
        noise_level = random.randint(noise_range[0], noise_range[1])
        noise = np.random.normal(0, noise_level, image.shape)
        augmented_image = np.clip(image + noise, 0, 255).astype(np.uint8)

        new_filename = os.path.splitext(filename)[0] + "_augverynoise"
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