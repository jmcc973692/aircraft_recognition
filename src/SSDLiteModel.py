import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import nms


class SSDLiteModel(torch.nn.Module):
    def __init__(self, num_classes, detection_threshold=0.2, iou_threshold=0.3, device=None):
        """
        Initializes the SSD model with specified number of classes, detection threshold, and IoU threshold for NMS.

        :param num_classes: Total number of classes (including background).
        :param detection_threshold: Minimum score required to consider a detection valid.
        :param iou_threshold: IoU threshold for non-maximum suppression.
        :param device: PyTorch device to use (cuda or cpu).
        """
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.detection_threshold = detection_threshold
        self.iou_threshold = iou_threshold  # IoU threshold for NMS
        self.model = self._load_model(num_classes)  # Load the base model

    def _load_model(self, num_classes):
        model = ssdlite320_mobilenet_v3_large(
            num_classes=num_classes, weights_backbone="DEFAULT", trainable_backbone_layers=3
        )
        return model.to(self.device)

    def forward(self, images, targets=None):
        if targets is not None:
            # Compute loss for training
            outputs = self.model(images, targets)
            return outputs

        # Perform detection for inference
        outputs = self.model(images)

        # Apply detection threshold and NMS
        filtered_outputs = []
        for output in outputs:
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]

            # Filter out boxes with low confidence scores
            high_confidence_idxs = scores > self.detection_threshold
            boxes = boxes[high_confidence_idxs]
            scores = scores[high_confidence_idxs]
            labels = labels[high_confidence_idxs]

            # Apply non-maximum suppression to reduce overlapping boxes
            keep_idxs = nms(boxes, scores, self.iou_threshold)

            filtered_outputs.append(
                {"boxes": boxes[keep_idxs], "scores": scores[keep_idxs], "labels": labels[keep_idxs]}
            )

        return filtered_outputs
