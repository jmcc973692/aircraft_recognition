import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, transform=None, test_mode=False):
        """
        Args:
            img_dir (string): Directory with all the images.
            csv_file (string, optional): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on an image.
            test_mode (bool, optional): Indicator if the dataset is used for testing (no labels).
        """
        self.img_dir = img_dir
        self.transform = transform
        self.test_mode = test_mode

        if csv_file is not None:
            self.img_labels = pd.read_csv(csv_file)
        else:
            self.img_labels = None

    def __len__(self):
        if self.img_labels is not None:
            return len(self.img_labels)
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.img_labels is not None:
            filename = self.img_labels.iloc[idx, 0]
            # Ensure the filename includes the correct extension
            if not filename.lower().endswith(".jpg"):
                filename += ".jpg"
            img_name = os.path.join(self.img_dir, filename)
        else:
            img_name = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])

        image = Image.open(img_name).convert("RGB")  # Force convert image to RGB

        if self.transform:
            image = self.transform(image)

        if self.test_mode:
            return image
        else:
            label = self.img_labels.iloc[idx, 1:].astype(float)
            label = torch.tensor(label.values).float()
            return image, label
