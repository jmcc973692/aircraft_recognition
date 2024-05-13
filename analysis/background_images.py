import os
import random

import pandas as pd
from PIL import Image


def load_bounding_boxes(csv_file):
    # Load bounding box data into a pandas DataFrame and add .jpg extension to filenames
    df = pd.read_csv(csv_file)
    df["filename"] = df["filename"] + ".jpg"
    return df


def is_within_bounding_boxes(x, y, width, height, bounding_boxes):
    # Check if the crop intersects any bounding box
    for _, bbox in bounding_boxes.iterrows():
        if x < bbox["xmax"] and x + width > bbox["xmin"] and y < bbox["ymax"] and y + height > bbox["ymin"]:
            return True
    return False


def generate_background_images(source_dir, bbox_data, target_dir, num_images=1500):
    os.makedirs(target_dir, exist_ok=True)
    image_files = [f for f in os.listdir(source_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    images_generated = 0

    while images_generated < num_images:
        img_file = random.choice(image_files)
        img_path = os.path.join(source_dir, img_file)
        # Use the filename with extension for matching
        bounding_boxes = bbox_data[bbox_data["filename"] == img_file]

        with Image.open(img_path) as img:
            for _ in range(10):  # Try up to 10 times per image to find a valid crop
                crop_width = random.randint(250, 600)
                crop_height = random.randint(250, 600)
                x = random.randint(0, max(0, img.width - crop_width))
                y = random.randint(0, max(0, img.height - crop_height))

                if not is_within_bounding_boxes(x, y, crop_width, crop_height, bounding_boxes):
                    crop = img.crop((x, y, x + crop_width, y + crop_height))
                    # Convert the image to RGB if it's not already
                    if crop.mode != "RGB":
                        crop = crop.convert("RGB")

                    crop.save(os.path.join(target_dir, f"img_back_{images_generated + 1}.jpg"))
                    images_generated += 1
                    break


# Usage
bbox_file = "input/aggregated_bounding_boxes_dataset_train.csv"
source_directory = "input/train_images"
target_directory = "input/crop/xBackground"
bounding_box_data = load_bounding_boxes(bbox_file)

generate_background_images(source_directory, bounding_box_data, target_directory)
