import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import natsort


class FolderDataset(Dataset):
    def __init__(self, main_dir, model):
        self.main_dir = main_dir
        self.model = model

        imgs = os.listdir(os.path.join(main_dir, 'imgs'))
        self.imgs = natsort.natsorted(imgs)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_file = os.path.join(self.main_dir, 'imgs', self.imgs[idx])
        gt_file = img_file.replace('imgs', 'gt')

        image = Image.open(img_file).convert("RGB")
        tensor_image = self.transform(image)

        depth = Image.open(gt_file)
        tensor_depth = self.transform(depth)

        tensor_depth = tensor_depth[0]

        return tensor_image, tensor_depth