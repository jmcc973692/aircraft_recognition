from typing import Tuple, Union

import torch
from PIL import Image
from torchvision import transforms


class CroppedImageTransformer:
    def __init__(self, resize_dims: Tuple[int, int] = (380, 380), augment: bool = False):
        """
        Initialize the transformer with resize dimensions and whether to apply augmentation.

        Args:
        resize_dims (Tuple[int, int]): Target dimensions (width, height) for resizing the images.
        augment (bool): Flag to determine whether to apply augmentation.
        """
        self.resize_dims = resize_dims
        self.augment = augment
        # Normalize with pre-defined mean and std values for EfficientNetB4.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Basic transformations: resize, convert to tensor, and normalize.
        self.base_transform = transforms.Compose(
            [transforms.Resize(self.resize_dims, interpolation=Image.BILINEAR), transforms.ToTensor(), self.normalize]
        )

        # Augmented transformations: color jittering, random flipping, random rotation, followed by basic transformations.
        self.augment_transform = transforms.Compose(
            [
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                self.base_transform,
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply the transformation to the given image.

        Args:
        img (Image.Image): The input PIL image to be transformed.

        Returns:
        torch.Tensor: The transformed image as a tensor.
        """
        # Padding if necessary to ensure the image has minimum dimensions.
        if img.width < self.resize_dims[0] or img.height < self.resize_dims[1]:
            padding = [
                (self.resize_dims[0] - img.width) // 2,  # left padding
                (self.resize_dims[1] - img.height) // 2,  # top padding
                (self.resize_dims[0] - img.width + 1) // 2,  # right padding
                (self.resize_dims[1] - img.height + 1) // 2,  # bottom padding
            ]
            img = transforms.Pad(padding, fill=0, padding_mode="constant")(img)

        # Apply augmented or base transformation based on the `augment` flag.
        if self.augment:
            return self.augment_transform(img)
        else:
            return self.base_transform(img)
