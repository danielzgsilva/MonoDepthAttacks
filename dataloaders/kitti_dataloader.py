__author__ = 'danielzgsilva'

import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Kittiloader(object):
    """
    param kittiDir: KITTI dataset root path, e.g. ~/data/KITTI/
    param mode: 'train' or 'test'
    param cam: camera id. 2 represents the left cam, 3 represents the right one
    """

    def __init__(self, kittiDir, mode, cam=2):
        self.mode = mode
        self.cam = cam
        self.files = []
        self.shared_idx = []
        self.kitti_root = kittiDir

        # read filenames files
        currpath = os.path.dirname(os.path.realpath(__file__))
        filepath = currpath + '/filenames/eigen_{}_files.txt'.format(self.mode)
        shared_path = currpath + '/filenames/eigen692_652_shared_index.txt'

        with open(filepath, 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_info = data.split(' ')

                assert cam == 2 or cam == 3, "Panic::Param 'cam' should be 2 or 3"
                data_idx_select = (0, 1)[cam == 3]

                self.files.append({
                    "rgb": data_info[data_idx_select],
                    "depth": data_info[data_idx_select+2]
                })

        print('found {} {} images for kitti'.format(len(self.files), mode))

        with open(shared_path, 'r') as f:
            shared_list = f.read().split('\n')
            for item in shared_list:
                if len(item) == 0:
                    continue
                self.shared_idx.append(int(item))

    def shared_index(self):
        return self.shared_idx

    def data_length(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.kitti_root, filename)
        assert os.path.exists(file_path), err_info + file_path
        return file_path

    def _read_depth(self, depth_path):
        # (copy from kitti devkit)
        # loads depth map D from png file
        # and returns it as a numpy array,

        depth_png = np.array(Image.open(depth_path), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float32) / 256.
        #depth[depth_png == 0] = -1.
        return depth

    def _read_data(self, item_files):
        rgb_path = self._check_path(
            item_files['rgb'], "Cannot find RGB Image ")
        depth_path = self._check_path(
            item_files['depth'], "Cannot find depth file ")

        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        depth = self._read_depth(depth_path)

        return rgb, depth

    def load_item(self, idx):
        """
        load an item for training or test
        interp_method can be selected from ['nop', 'linear', 'nyu']
        """
        item_files = self.files[idx]
        rgb, depth = self._read_data(item_files)
        return rgb, depth


class KITTIDataset(Dataset):
    def __init__(self, root, type, model='resnet'):
        self.root = root
        self.type = type
        self.model = model

        if self.model == 'dpt':
            self.output_size = (352, 1216)
        else:
            self.output_size = (228, 912)

        if type == 'train':
            self.transform = self.train_transform
        elif type == 'test':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                                                  "Supported dataset types are: train, val"))

        # use left image by default
        self.kittiloader = Kittiloader(root, type, cam=2)

    def train_transform(self, rgb, depth):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=5),
            transforms.CenterCrop(self.output_size),
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor()
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor

    def val_transform(self, rgb, depth):
        if self.model == 'dpt':
            height, width, _ = rgb.shape
            top = height - 352
            left = (width - 1216) // 2
            rgb = rgb[top: top + 352, left: left + 1216, :]

            height, width = depth.shape
            top = height - 352
            left = (width - 1216) // 2
            depth = depth[top: top + 352, left: left + 1216]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])

        rgb_tensor = transform(rgb)
        depth_tensor = transform(depth)

        return rgb_tensor, depth_tensor

    def __getitem__(self, idx):
        # load an item according to the given index
        rgb, depth = self.kittiloader.load_item(idx)

        if self.transform is not None:
            rgb_tensor, depth_tensor = self.transform(rgb, depth)
        else:
            raise RuntimeError("transform not defined")

        return rgb_tensor, depth_tensor

    def __len__(self):
        return self.kittiloader.data_length()
