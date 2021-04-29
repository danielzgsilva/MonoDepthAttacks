__author__ = 'danielzgsilva'

import numpy as np
from dataloaders.dataloader import MyDataloader

from torchvision import transforms

iheight, iwidth = 480, 640  # raw image size


class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', model='resnet'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        if model == 'dpt':
            self.output_size = (480, 640)
        else:
            self.output_size = (228, 304)

    def train_transform(self, rgb, depth):
        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10),
            transforms.CenterCrop(self.output_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor

    def val_transform(self, rgb, depth):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor
