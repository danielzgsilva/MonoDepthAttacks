import numpy as np
# import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

from torchvision import transforms

iheight, iwidth = 480, 640  # raw image size


class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (228, 304)

    def train_transform(self, rgb, depth):
        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=5),
            transforms.CenterCrop(self.output_size),
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor

    def val_transform(self, rgb, depth):
        transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.CenterCrop(self.output_size),
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor
