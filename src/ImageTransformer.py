from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image


class ImageTransformer:
    def __init__(self, resize_dims=(320, 320), mean=None, std=None):
        """
        Initialize the transformer with resizing, normalization parameters, and new dimensions.

        Args:
            resize_dims (tuple of int): Dimensions to resize images to.
            mean (list of float): The mean values for normalization.
            std (list of float): The standard deviation values for normalization.
        """
        self.resize_dims = resize_dims
        self.mean = mean or [0.485, 0.456, 0.406]  # Default mean for COCO dataset
        self.std = std or [0.229, 0.224, 0.225]  # Default std for COCO dataset

    def _adjust_bounding_boxes(self, resized_image, target):
        """
        Adjust bounding boxes according to the image's new dimensions.
        """
        orig_width, orig_height = target["original_dims"]
        new_width, new_height = resized_image.size
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        new_boxes = target["boxes"].clone().float()
        new_boxes[:, 0] *= scale_x
        new_boxes[:, 1] *= scale_y
        new_boxes[:, 2] *= scale_x
        new_boxes[:, 3] *= scale_y
        target["boxes"] = new_boxes.long()  # Convert back to long for compatibility with PyTorch

        del target["original_dims"]  # Remove 'original_dims' from target
        return target

    def __call__(self, image: Image.Image, target: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Transform the image and optionally handle the target.

        Args:
            image (PIL.Image): The image to transform.
            target (dict, optional): Target dictionary with bounding boxes and labels.

        Returns:
            tuple: Transformed tensor and the optional target dictionary.
        """
        resized_image = F.resize(image, self.resize_dims)
        image_tensor = F.to_tensor(resized_image)
        normalized_image = F.normalize(image_tensor, mean=self.mean, std=self.std)

        if target:
            target["original_dims"] = image.size
            target = self._adjust_bounding_boxes(resized_image, target)
        return (normalized_image, target) if target else normalized_image
