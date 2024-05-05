import os

from PIL import Image
from torch.utils.data import Dataset


class CroppedDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images organized by class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(os.listdir(img_dir))  # List of classes based on folder names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []  # List of tuples (image_path, class_index)

        # Load all image file paths and their class labels
        for class_name in self.classes:
            class_dir = os.path.join(self.img_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(class_dir, img_file), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_index = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Open and convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, class_index
