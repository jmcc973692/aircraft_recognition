import numpy as np
import pytest
import torch
import torchvision.transforms.functional as F
from PIL import Image

from src.ImageTransformer import ImageTransformer


@pytest.fixture
def transformer():
    # Create a transformer with specific resizing and default ImageNet mean/std
    return ImageTransformer(resize_min=800, resize_max=1333)


@pytest.fixture
def dummy_image():
    # Create a sample image with random data
    return Image.fromarray(np.uint8(np.random.rand(1000, 800, 3) * 255))


def test_image_transformation(transformer, dummy_image):
    """
    Test if the ImageTransformer correctly resizes and normalizes images.
    """
    # Transform the dummy image with the transformer
    transformed_result = transformer(dummy_image)

    # Determine if the transformer returned a single item or a tuple
    if isinstance(transformed_result, tuple):
        transformed_image = transformed_result[0]
    else:
        transformed_image = transformed_result

    # Expected resizing using the transformer's logic
    resized_image = transformer.resize_with_aspect(dummy_image)

    # Convert the resized image to a tensor
    original_tensor = F.to_tensor(resized_image)

    # Expected mean and std for ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Manually normalize the resized image
    manually_normalized = (original_tensor - mean[:, None, None]) / std[:, None, None]

    # Check if the transformed image matches the manually normalized image
    assert torch.allclose(
        transformed_image, manually_normalized, atol=0.01
    ), "Transformed image should match manually normalized image."
