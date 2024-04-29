import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class SSDModel(torch.nn.Module):
    def __init__(self, num_classes, detection_threshold=0.5, device=None):
        """
        Initializes the SSD model with a specified number of classes and a detection threshold.

        :param num_classes: Total number of classes (including background).
        :param detection_threshold: The minimum score required to consider a detection valid.
        :param device: PyTorch device to use (cuda or cpu).
        """
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.detection_threshold = detection_threshold  # Set the detection threshold
        self.model = self._load_model(num_classes)  # Load the base model

    def _load_model(self, num_classes):
        """
        Loads the SSDLite320 model with a specified number of classes.
        """
        model = ssdlite320_mobilenet_v3_large(
            num_classes=num_classes, weights_backbone="DEFAULT", trainable_backbone_layers=0
        )
        return model.to(self.device)

    def forward(self, images, targets=None):
        """
        Forward pass for the model, applying a detection threshold.

        :param images: The images to be processed.
        :param targets: Optional target data, used for computing loss during training.
        :return: Outputs or loss_dict depending on mode.
        """
        if targets is not None:
            # If targets are provided, compute loss for training
            outputs = self.model(images, targets)  # Forward pass with targets
            return outputs  # Loss dictionary for training

        # Perform detection for inference
        outputs = self.model(images)  # Forward pass without targets

        # Apply detection threshold to filter low-confidence detections
        filtered_outputs = []
        for output in outputs:
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]

            # Get indices of boxes with scores above the threshold
            high_confidence_indices = torch.where(scores > self.detection_threshold)[0]

            # Filter boxes, scores, and labels based on high confidence
            filtered_boxes = boxes[high_confidence_indices]
            filtered_scores = scores[high_confidence_indices]
            filtered_labels = labels[high_confidence_indices]

            filtered_outputs.append({"boxes": filtered_boxes, "scores": filtered_scores, "labels": filtered_labels})

        return filtered_outputs  # Return filtered detections
