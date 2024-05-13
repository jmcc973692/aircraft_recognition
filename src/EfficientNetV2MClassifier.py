from functools import partial

import torch
from torch import nn
from torchvision.models.efficientnet import EfficientNet, EfficientNet_V2_M_Weights, _efficientnet_conf


class EfficientNetV2MClassifier(torch.nn.Module):
    def __init__(self, num_classes=16, device=None):
        super(EfficientNetV2MClassifier, self).__init__()
        # Load a pre-trained EfficientNet-B7 model
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
        weights = EfficientNet_V2_M_Weights.DEFAULT
        norm_layer = partial(torch.nn.BatchNorm2d, eps=1e-03)
        dropout = 0.3
        self.model = EfficientNet(
            inverted_residual_setting=inverted_residual_setting,
            dropout=dropout,
            norm_layer=norm_layer,
            last_channel=last_channel,
        )
        self.model.load_state_dict(weights.get_state_dict())
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move the model to the specified device
        self.to(self.device)

    def forward(self, x):
        # Forward pass through the model
        x = self.model(x)
        # Apply sigmoid to output to get probabilities for each class
        return torch.sigmoid(x)
