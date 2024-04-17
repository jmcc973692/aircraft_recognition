import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedTorchConvNet(nn.Module):
    def __init__(self):
        super(OptimizedTorchConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # Reduce size more quickly
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)  # Further reduce size
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Linear(128, 15)  # Only one FC layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)  # Global average pooling replaces flattening
        x = x.view(x.size(0), -1)  # Flatten to vector
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)
