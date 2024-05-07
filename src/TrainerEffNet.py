import torch
import torch.nn as nn


class TrainerEffNet:
    def __init__(self, model, device, optimizer):
        """
        Initialize the Trainer with an EfficientNet model, device, and optimizer.
        Args:
            model (torch.nn.Module): The EfficientNet model to train.
            device (torch.device): The device to use for training (CPU/GPU).
            optimizer (torch.optim.Optimizer): The optimizer for training.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = nn.BCELoss()  # Handles sigmoid activation + binary cross-entropy loss.

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch using the training data loader.
        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_epoch(self, valid_loader):
        """
        Validate the model for one epoch using the validation data loader.
        Args:
            valid_loader (torch.utils.data.DataLoader): The validation data loader.
        Returns:
            float: The average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(valid_loader)
