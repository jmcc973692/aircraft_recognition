import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ImageDataset import ImageDataset


def compute_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize the images if necessary
            transforms.ToTensor(),  # Convert images to tensor
        ]
    )

    dataset = ImageDataset(img_dir="./input/train_images/", transform=transform, test_mode=True)
    loader = DataLoader(dataset, batch_size=10, num_workers=4, shuffle=False)

    mean, std = compute_mean_std(loader)
    print(f"Mean: {mean}")
    print(f"Std Deviation: {std}")


if __name__ == "__main__":
    main()
