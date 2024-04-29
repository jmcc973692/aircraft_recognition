import os

import pandas as pd
import pytest
import torch
import torchvision.transforms.functional as F
from PIL import Image

from src.AircraftBoundingDataset import AircraftBoundingDataset
from src.ImageTransformer import ImageTransformer


# Define a fixture for initializing the dataset
@pytest.fixture
def aircraft_bounding_dataset():
    csv_file = "test/test_bounding_box_labels.csv"
    img_dir = "test/test_images/"
    transform = ImageTransformer()  # Ensure transformation logic is consistent
    return AircraftBoundingDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)


# Test the dataset initialization
def test_dataset_initialization(aircraft_bounding_dataset):
    assert len(aircraft_bounding_dataset) > 0, "Dataset should have some data."


# Test fetching an item from the dataset
def test_get_item(aircraft_bounding_dataset):
    # Get the first item from the dataset
    image, target = aircraft_bounding_dataset[0]

    # Check if the image is a tensor
    assert isinstance(image, torch.Tensor), "Image should be a tensor."

    # Check if the target has the required keys
    assert "boxes" in target, "Target should have 'boxes'."
    assert "labels" in target, "Target should have 'labels'."

    # Check if the bounding box format is correct (should be [N, 4])
    assert target["boxes"].shape[1] == 4, "Bounding boxes should be of shape [N, 4]."

    # Check if all labels are set to 1
    assert torch.all(target["labels"] == 1), "All labels should be set to 1."


# Test the transformations
def test_transform(aircraft_bounding_dataset):
    """
    Test if the dataset's transformation matches manual normalization and resizing to 320x320.
    """
    # Get the first image and target from the dataset
    image, target = aircraft_bounding_dataset[0]

    # Default mean and std for SSDLite models trained on COCO
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Manually normalize and resize the original image to 320x320
    original_image = Image.open("test/test_images/img_4.jpg").convert("RGB")
    resized_image = F.resize(original_image, (320, 320))
    original_tensor = F.to_tensor(resized_image)

    manually_normalized = F.normalize(original_tensor, mean=mean, std=std)

    # Check if the dataset's transformation matches the manual normalization
    tolerance = 0.05  # Allow some tolerance due to floating-point operations
    assert torch.allclose(
        image, manually_normalized, atol=tolerance
    ), "Dataset result should match manual normalization and resizing to 320x320."

    # Check if bounding boxes have been adjusted properly
    orig_width, orig_height = original_image.size  # Original image dimensions
    new_width, new_height = 320, 320  # Resized dimensions

    # Calculate the expected scale factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    original_box = torch.tensor([[1, 1759, 6016, 4016]])
    expected_boxes = original_box.clone().float()
    expected_boxes[:, 0] *= scale_x  # xmin
    expected_boxes[:, 1] *= scale_y  # ymin
    expected_boxes[:, 2] *= scale_x  # xmax
    expected_boxes[:, 3] *= scale_y  # ymax

    # Check if bounding boxes were transformed correctly
    assert torch.equal(expected_boxes.long(), target["boxes"]), "Bounding boxes should be adjusted correctly."


# Test handling multiple bounding boxes
def test_multiple_bounding_boxes(aircraft_bounding_dataset):
    """
    Test if the dataset handles images with multiple bounding boxes correctly.
    """
    # Get an image with multiple bounding boxes
    image, target = aircraft_bounding_dataset[2]  # Index should point to an image with multiple bounding boxes

    # Expected number of bounding boxes
    expected_box_count = 2  # Adjust based on your specific test dataset

    # Check if the image is a tensor
    assert isinstance(image, torch.Tensor), "Image should be a tensor."

    # Check if the bounding box format is correct
    assert target["boxes"].shape[1] == 4, "Bounding boxes should be of shape [N, 4]."

    # Check the number of bounding boxes
    assert (
        target["boxes"].shape[0] == expected_box_count
    ), f"Expected {expected_box_count} bounding boxes, but got {target['boxes'].shape[0]}."

    # Validate that all labels are 1
    assert torch.all(target["labels"] == 1), "All labels should be set to 1 for aircraft detection."
