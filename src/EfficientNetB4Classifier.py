import timm
import torch
import torch.nn as nn
from timm.models.efficientnet import default_cfgs


class EfficientNetB4Classifier(torch.nn.Module):
    def __init__(self, num_classes=16, pretrained=True, device=None):
        super(EfficientNetB4Classifier, self).__init__()
        # Load a pre-trained EfficientNet-B7 model
        self.model = timm.create_model(
            model_name="efficientnet_b4", drop_rate=0.2, pretrained=pretrained, num_classes=num_classes
        )
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move the model to the specified device
        self.to(self.device)

    def forward(self, x):
        # Forward pass through the model
        x = self.model(x)
        # Apply sigmoid to output to get probabilities for each class
        return torch.sigmoid(x)
