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
        for index, row in self.annotations.iterrows():
            self.image_groups[row["filename"]].append(row)

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
        image_data = self.image_groups[image_key]
        img_path = os.path.join(self.img_dir, f"{image_key}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Aggregate all bounding boxes and labels for the image
        boxes = []
        labels = []
        for data in image_data:
            boxes.append([data["xmin"], data["ymin"], data["xmax"], data["ymax"]])
            labels.append(1)  # Assuming label '1' for all aircraft

        # Convert boxes and labels to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
