from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
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

        # Define the basic transformations
        self.base_transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.resize_dims[1],
                    min_width=self.resize_dims[0],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                ),
                A.Resize(self.resize_dims[0], self.resize_dims[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        # Define augmented transformations
        if self.augment:
            self.augment_transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.08,
                        scale_limit=0.15,
                        rotate_limit=20,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    ),
                    A.PadIfNeeded(
                        min_height=self.resize_dims[1],
                        min_width=self.resize_dims[0],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.GaussNoise(p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.ColorJitter(p=0.4),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.4),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
                    A.Perspective(scale=(0.05, 0.1), p=0.2),
                    A.GridDistortion(p=0.2),
                    A.RandomGridShuffle(grid=(3, 3), p=0.2),
                    A.Resize(self.resize_dims[0], self.resize_dims[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.augment_transform = self.base_transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply the transformation to the given image.

        Args:
        img (Image.Image): The input PIL image to be transformed.

        Returns:
        torch.Tensor: The transformed image as a tensor.
        """
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Apply augmented or base transformation based on the `augment` flag.
        if self.augment:
            augmented = self.augment_transform(image=img_np)
        else:
            augmented = self.base_transform(image=img_np)

        return augmented["image"]
