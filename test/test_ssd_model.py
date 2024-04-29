import pytest
import torch

from src.SSDModel import SSDModel


@pytest.fixture
def ssd_model():
    # Creates an instance of SSDModel with 16 classes (including background)
    return SSDModel(num_classes=16, device="cuda" if torch.cuda.is_available() else "cpu")


def test_model_initialization(ssd_model):
    """
    Test if the SSD model is correctly initialized.
    """
    assert ssd_model.model is not None, "Model should be initialized"
    assert isinstance(ssd_model.model, torch.nn.Module), "Model should be a PyTorch module"
