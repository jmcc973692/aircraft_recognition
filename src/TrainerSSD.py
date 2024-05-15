import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert


class TrainerSSD:
    def __init__(self, model, device, optimizer):
        """
        Initialize the Trainer with an SSD model, device, and optimizer.

        Args:
            model (torch.nn.Module): The SSD model to train.
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
        total_loss = 0.0  # Accumulate loss for the epoch

        # Iterate through the training data
        for images, targets in train_loader:
            # Move images and targets to the appropriate device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in tgt.items()} for tgt in targets]

            self.optimizer.zero_grad()  # Reset gradients
            loss_dict = self.model(images, targets)  # Forward pass and compute losses
            losses = sum(loss_dict.values())  # Aggregate total loss

            # Backpropagation and optimization
            losses.backward()
            self.optimizer.step()

            # Accumulate the total loss for the epoch
            total_loss += losses.item()

        # Return the average loss for the epoch
        return total_loss / len(train_loader)

    def validate_epoch(self, valid_loader):
        self.model.eval()  # Ensure the model is in evaluation mode
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)  # Get model predictions

                # Prepare detections and ground truths for torchmetrics
                for i, output in enumerate(outputs):
                    pred_boxes = output["boxes"]
                    pred_labels = output["labels"]
                    pred_scores = output["scores"]

                    # Ground truth
                    gt_boxes = targets[i]["boxes"]
                    gt_labels = targets[i]["labels"]

                    # Update metric for each prediction and ground truth pair
                    metric.update(
                        preds=[{"boxes": pred_boxes.cpu(), "scores": pred_scores.cpu(), "labels": pred_labels.cpu()}],
                        target=[{"boxes": gt_boxes.cpu(), "labels": gt_labels.cpu()}],
                    )

        # Compute mAP
        result = metric.compute()
        mean_ap = result["map"].item()  # Overall mAP
        return -mean_ap
