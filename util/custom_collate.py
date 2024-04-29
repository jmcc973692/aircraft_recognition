import torch


def custom_collate_fn(batch):
    """
    Custom collate function to handle varying image and target sizes.
    """
    images = torch.stack([item[0] for item in batch])  # Ensures all images are tensors and stacks them
    targets = [item[1] for item in batch]  # Returns a list of target dictionaries

    return images, targets
