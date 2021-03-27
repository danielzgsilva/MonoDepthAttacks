import numpy as np
# import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

from torchvision import transforms


class KITTIDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(KITTIDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (228, 912)

    def train_transform(self, rgb, depth):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=5),
            transforms.CenterCrop(self.output_size),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.ToTensor()
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor

    def val_transform(self, rgb, depth):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor