import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.FasterRCNNAircraft import FasterRCNNAircraft


@pytest.fixture
def detector():
    # Path to the pretrained weights
    weights_path = "weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    # Setup the device to use for the test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize the detector with the local pretrained weights
    return FasterRCNNAircraft(num_classes=16, weights_path=weights_path, device=device)


def test_model_initialization(detector):
    """
    Test if the model is correctly initialized and modified.
    """
    # Assert the predictor modification happened correctly
    assert isinstance(detector.model.roi_heads.box_predictor, FastRCNNPredictor)


def test_predict(detector):
    """
    Test the predict function using actual model methods.
    """

    # Create a dummy image using PIL
    dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    # Perform prediction
    prediction = detector.predict([dummy_image])

    # Check if the output matches the expected format
    assert isinstance(prediction, list)
    assert all(isinstance(p, dict) for p in prediction)
    assert all("boxes" in p and "labels" in p and "scores" in p for p in prediction)
