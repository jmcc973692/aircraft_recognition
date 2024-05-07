import torch
from PIL import Image
from torch.utils.data import Dataset


class CroppedDataset(Dataset):
    def __init__(self, images, labels, transformer=None):
        """
        Initialize the dataset with a list of image paths and labels.

        Args:
        images (list of str): List of paths to the images. Each image is associated with one label.
        labels (list of int): List of labels corresponding to the images. Each label is an integer
                              representing the class index of the image.
        transformer (callable, optional): Transformer to apply to the images for preprocessing
                                          (e.g., resizing, normalization). Default is None.
        """
        self.images = images
        self.labels = labels
        self.transformer = transformer
        self.num_classes = 15

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image and its one-hot encoded label based on the index.

        Args:
        idx (int): Index of the image and label to retrieve.

        Returns:
        tuple: A tuple containing the transformed image and its one-hot encoded label.
               The label is returned as a float tensor of shape (num_classes,) with a 1.0 in the
               position corresponding to the class of the image and 0.0 elsewhere.
        """
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformer if provided
        if self.transformer:
            image = self.transformer(image)

        # Convert label to one-hot encoding
        one_hot_label = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot_label[label] = 1.0  # Set the index of the class label to 1

        return image, one_hot_label
