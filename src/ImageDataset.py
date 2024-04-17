import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, transform=None, test_mode=False):
        """
        Args:
            img_dir (string): Directory with all the images.
            csv_file (string, optional): Path to the csv file with annotations.
            transform (callable, optional): Transform to be applied on an image.
            test_mode (bool, optional): Indicator if the dataset is used for testing (no labels).
        """
        self.img_dir = img_dir
        self.transform = transform
        self.test_mode = test_mode
        self.images = []
        self.labels = []

        # Load image labels if available
        if csv_file is not None:
            self.df_labels = pd.read_csv(csv_file)
        else:
            self.df_labels = pd.DataFrame()

        # Preload all images and apply transformations
        self.preload_images()

        # Stack all images into a single tensor
        self.images = torch.stack(self.images)

    def preload_images(self):
        # Load all filenames from the directory if no labels provided
        if self.df_labels.empty:
            file_names = os.listdir(self.img_dir)
        else:
            # Use filenames from CSV for label matching
            file_names = self.df_labels.iloc[:, 0].tolist()

        for filename in file_names:
            full_filename = filename
            # Append .jpg if it's not already there to open the file
            if not full_filename.lower().endswith(".jpg"):
                full_filename += ".jpg"

            image_path = os.path.join(self.img_dir, full_filename)
            try:
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)

                self.images.append(image)  # Make sure transform includes ToTensor()

                # Accessing labels using filename without .jpg extension
                if not self.test_mode and not self.df_labels.empty:
                    label_data = self.df_labels[self.df_labels.iloc[:, 0] == filename]
                    if not label_data.empty:
                        label = label_data.iloc[0, 1:].astype(float).values
                        self.labels.append(torch.tensor(label).float())
                    else:
                        # Log if no label found for the filename
                        print(f"No label found for {filename}, using default label or skipping.")
                        # Handle the case for no labels found: e.g., skip or use default label
                        self.images.pop()  # Optional: Remove the last image if skipping
            except FileNotFoundError:
                print(f"File not found: {image_path}")
                continue  # Skip this file if not found

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.images[idx]
        else:
            label = self.labels[idx]
            return self.images[idx], label
