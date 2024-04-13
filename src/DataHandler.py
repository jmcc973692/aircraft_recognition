from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .ImageDataset import ImageDataset


class DataHandler:
    def __init__(self, img_dir, transform, batch_size, num_workers, pin_memory):
        self.transform = transform
        self.dataset = ImageDataset(
            csv_file=f"{img_dir}/train.csv", img_dir=f"{img_dir}/train_images", transform=self.transform
        )
        train_size = int(0.8 * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        train_dataset, valid_dataset = random_split(self.dataset, [train_size, valid_size])
        self.trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
        )
        self.validloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers
        )

    @staticmethod
    def setup_transforms():
        return transforms.Compose(
            [
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4875, 0.5251, 0.5581], std=[0.1817, 0.1764, 0.1857]),
            ]
        )

    def setup_test_loader(self, img_dir, batch_size, num_workers, pin_memory):
        test_transform = transforms.Compose(
            [
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4875, 0.5251, 0.5581], std=[0.1817, 0.1764, 0.1857]),
            ]
        )
        test_dataset = ImageDataset(img_dir=img_dir, transform=test_transform, test_mode=True)
        return DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers
        )
