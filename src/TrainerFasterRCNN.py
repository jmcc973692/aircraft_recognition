import torch


class TrainerFasterRCNN:
    def __init__(self, model, device, optimizer):
        """
        Initialize the Trainer with a Faster R-CNN model, device, and optimizer.

        Args:
            model (torch.nn.Module): The Faster R-CNN model to train.
            device (torch.device): The device to use for training (CPU/GPU).
            optimizer (torch.optim.Optimizer): The optimizer for training.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()  # Set the model to training mode
        total_loss = 0  # Accumulate loss for the epoch

        # Iterate through the training data
        for images, targets in train_loader:
            images = [img.to(self.device) for img in images]  # Move images to the device
            targets = [{k: v.to(self.device) for k, v in tgt.items()} for tgt in targets]  # Move targets to the device

            self.optimizer.zero_grad()  # Reset gradients
            loss_dict = self.model(images, targets)  # Forward pass and compute losses
            losses = sum(loss for loss in loss_dict.values())  # Aggregate total loss

            # Backpropagation and optimization
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()  # Accumulate the total loss

        # Return the average loss for the epoch
        return total_loss / len(train_loader)

    def validate_epoch(self, valid_loader):
        """
        Validate the model for one epoch.

        Args:
            valid_loader (torch.utils.data.DataLoader): The validation data loader.

        Returns:
            float: The average validation loss for the epoch.
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0  # Accumulate loss for the epoch

        with torch.no_grad():  # No gradients during validation
            for images, targets in valid_loader:
                images = [img.to(self.device) for img in images]  # Move images to the device
                targets = [
                    {k: v.to(self.device) for k, v in tgt.items()} for tgt in targets
                ]  # Move targets to the device

                # Compute the losses without backpropagation
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()  # Accumulate the total loss

        # Return the average validation loss for the epoch
        return total_loss / len(valid_loader)
