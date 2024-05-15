import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image


class ImageTransformerAugment:

    def __init__(self, resize_dims=(320, 320), mean=None, std=None):
        self.resize_dims = resize_dims
        self.mean = mean or [0.485, 0.456, 0.406]  # Default mean for COCO dataset
        self.std = std or [0.229, 0.224, 0.225]  # Default std for COCO dataset

        self.basic_transforms = A.Compose(
            [
                A.Resize(self.resize_dims[0], self.resize_dims[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        self.augmentations = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(height=320, width=320, scale=(0.08, 1.0), ratio=(0.75, 1.33), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
                A.ColorJitter(p=0.4),
                A.GaussianBlur(p=0.1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.3),
                A.RandomRain(drop_length=3, blur_value=1, p=0.2),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.2),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.3
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_area=2, min_visibility=0.05),
        )

    def __call__(self, image, target):
        image_np = np.array(image)  # Convert PIL Image to numpy array for Albumentations
        augmented = self.augmentations(image=image_np, bboxes=target["boxes"].numpy(), labels=target["labels"].numpy())

        if not augmented["bboxes"]:
            transformed = self.basic_transforms(
                image=image_np, bboxes=target["boxes"].numpy(), labels=target["labels"].numpy()
            )
        else:
            image_tensor = torch.from_numpy(np.array(augmented["image"])).permute(
                2, 0, 1
            )  # For using dimensions in clipping
            clipped_bboxes = self.clip_bboxes_to_image(
                augmented["bboxes"], image_tensor.shape[1], image_tensor.shape[2]
            )
            transformed = self.basic_transforms(
                image=augmented["image"], bboxes=clipped_bboxes, labels=augmented["labels"]
            )

        # Assuming 'transformed' directly manipulates and returns tensors if necessary
        target["boxes"] = torch.stack([torch.tensor(bbox, dtype=torch.float32) for bbox in transformed["bboxes"]])
        target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)

        return transformed["image"], target

    @staticmethod
    def clip_bboxes_to_image(bboxes, img_height, img_width):
        """
        Adjusts bounding box coordinates so that they fit within the image frame.
        """
        clipped_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0, min(x_min, img_width))
            x_max = max(0, min(x_max, img_width))
            y_min = max(0, min(y_min, img_height))
            y_max = max(0, min(y_max, img_height))
            if x_max > x_min and y_max > y_min:  # Ensure the bbox is valid
                clipped_bboxes.append([x_min, y_min, x_max, y_max])
        return clipped_bboxes
