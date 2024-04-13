import pytest
import torch
from torchvision import transforms

from src.ImageDataset import ImageDataset


@pytest.fixture
def setup_image_dataset():
    # Transformation to apply to each image
    transform = transforms.Compose(
        [transforms.Resize((50, 50)), transforms.ToTensor()]  # Resize to ensure consistency  # Convert to tensor
    )

    # Correct path to images and labels based on your project structure
    img_dir = "test/test_images"
    csv_file = "test/test_labels.csv"

    # Initialize dataset with multi-labels
    return ImageDataset(img_dir=img_dir, csv_file=csv_file, transform=transform, test_mode=False)


def test_loading_with_labels(setup_image_dataset):
    dataset = setup_image_dataset
    img, labels = dataset[0]  # Fetch the first sample

    assert img.size() == (3, 50, 50), "Image dimensions or channel are incorrect after transform"
    assert len(labels) == 15, "There should be 15 labels for each image"
    assert isinstance(labels, torch.Tensor), "Labels should be converted to a tensor"
    assert labels.dtype == torch.float32, "Label tensor should be of type float32"


@pytest.fixture
def setup_image_dataset_test_mode():
    transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])
    img_dir = "test/test_images"

    # Initialize dataset in test mode, where no labels are expected
    return ImageDataset(img_dir=img_dir, transform=transform, test_mode=True)


def test_loading_in_test_mode(setup_image_dataset_test_mode):
    dataset = setup_image_dataset_test_mode
    img = dataset[0]  # Fetch the first image without labels

    assert img.size() == (3, 50, 50), "Image dimensions or channels are incorrect"


def test_img_4_labels(setup_image_dataset):
    dataset = setup_image_dataset
    _, labels = dataset[0]  # img_4 should be the first image if sorted as in the CSV
    # Assuming the order of labels in the tensor follows the CSV header:
    # ['A10','B1','B2','B52','C130','C17','E2','F14','F15','F16','F18','F22','F35','F4','V22']
    expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).float()
    assert torch.equal(labels, expected), "Labels for img_4 are incorrect"


def test_img_5_labels(setup_image_dataset):
    dataset = setup_image_dataset
    _, labels = dataset[1]  # img_5 should be the second image
    expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).float()
    assert torch.equal(labels, expected), "Labels for img_5 are incorrect"


if __name__ == "__main__":
    pytest.main()
