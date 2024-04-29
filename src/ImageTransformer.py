from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image


class ImageTransformer:
    def __init__(self, mean=None, std=None):
        """
        Initialize the transformer with resizing and normalization parameters.

        Args:
            mean (list of float): The mean values for normalization (SSDLite defaults).
            std (list of float): The standard deviation values for normalization (SSDLite defaults).
        """
        # Default mean and std for SSDLite models trained on COCO
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.resize_dims = (320, 320)

    def _adjust_bounding_boxes(self, resized_image, target):
        """
        Adjust bounding boxes according to the image's new dimensions.
        """
        # Calculate the scale factors for x and y
        orig_width, orig_height = target["original_dims"]  # Store original dimensions if needed
        new_width, new_height = resized_image.size

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        # Adjust the bounding boxes
        new_boxes = target["boxes"].clone().float()  # Ensure it's in float format for operations
        new_boxes[:, 0] *= scale_x  # xmin
        new_boxes[:, 1] *= scale_y  # ymin
        new_boxes[:, 2] *= scale_x  # xmax
        new_boxes[:, 3] *= scale_y  # ymax

        # Cast back to Long if needed (or Integer)
        new_boxes = new_boxes.long()  # Explicit cast to integer (Long)

        target["boxes"] = new_boxes

        # Drop 'original_dims' from the dictionary
        del target["original_dims"]

        return target

    def __call__(self, image: Image.Image, target: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Transform the image, resize to 320x320, and adjust bounding boxes.

        Args:
            image (PIL.Image): The image to transform.
            target (dict, optional): Target dictionary with bounding boxes and labels.

        Returns:
            tuple: Transformed and resized tensor and the updated target.
        """
        # Resize the image to 320x320
        resized_image = F.resize(image, self.resize_dims)

        # Convert to tensor and normalize
        image_tensor = F.to_tensor(resized_image)
        normalized_image = F.normalize(image_tensor, mean=self.mean, std=self.std)

        if target:
            # Adjust bounding boxes after resizing
            target = self._adjust_bounding_boxes(resized_image, target)

        return (normalized_image, target) if target else (normalized_image, None)
