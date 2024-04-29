import os
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .ImageTransformer import ImageTransformer


class AircraftBoundingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Initialize the dataset with CSV annotations and the image directory.

        :param csv_file: Path to the CSV file containing annotations.
        :param img_dir: Directory where images are stored.
        :param transform: Optional transformation to be applied to the images.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or ImageTransformer()  # Default transformer if none provided

        # Group annotations by image filename
        self.image_groups = defaultdict(list)
        for _, row in self.annotations.iterrows():
            filename = row["filename"]
            self.image_groups[filename].append(row)

    def __len__(self):
        return len(self.image_groups)  # Number of unique images in the dataset

    def __getitem__(self, index):
        """
        Retrieve an image and its associated bounding boxes and labels.

        :param index: Index of the image in the dataset.
        :return: Tuple containing the transformed image and target (with bounding boxes and labels).
        """
        # Get the image key
        image_key = list(self.image_groups.keys())[index]
        img_name = os.path.join(self.img_dir, f"{image_key}.jpg")
        image = Image.open(img_name).convert("RGB")

        # Gather annotations for this image
        annotations = self.image_groups[image_key]

        # Create bounding boxes and a single label (1 for aircraft)
        boxes = []
        for annotation in annotations:
            xmin = annotation["xmin"]
            ymin = annotation["ymin"]
            xmax = annotation["xmax"]
            ymax = annotation["ymax"]

            # Validate bounding box coordinates
            if xmin >= xmax or ymin >= ymax:
                raise ValueError(f"Invalid bounding box coordinates: {xmin, ymin, xmax, ymax}")

            # Add bounding box to the list
            boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))

        # Convert the list of boxes into a tensor
        boxes = torch.stack(boxes)

        # Use a single label for all aircraft (1)
        labels = torch.tensor([1] * len(boxes))  # All aircraft are labeled as 1

        target = {"boxes": boxes, "labels": labels, "original_dims": image.size}

        # Apply any transformation (e.g., resizing and normalizing)
        if self.transform:
            image, target = self.transform(image, target)

        return image, target
