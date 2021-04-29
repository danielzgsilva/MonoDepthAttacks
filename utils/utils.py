import glob
import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataloaders import kitti_dataloader, nyu_dataloader, folder_loader
from dataloaders.path import Path

cmap = plt.cm.get_cmap('binary_r')


def parse_command():
    modality_names = ['rgb', 'rgbd', 'd']

    import argparse
    parser = argparse.ArgumentParser(description='FCRN')
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--attack', '-a', default=None, type=str)
    parser.add_argument('--adv_training', default=False,
                        type=bool, help='perform adversarial training')
    parser.add_argument('--eval_output_dir', default=None, type=str)

    parser.add_argument('--num_images_to_save', default=100, type=int)
    parser.add_argument('--save_image_dir', default=None, type=str)

    parser.add_argument('--decoder', default='upproj', type=str)
    parser.add_argument('--resnet_layers', default=50, type=int)
    parser.add_argument('--resume',
                        default=None,
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('-b', '--batch-size', default=1,
                        type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--loss', default='l1', type=str)
    parser.add_argument('--epochs', default=4, type=int, metavar='N',
                        help='number of total epochs to run (default: 4)')
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')
    parser.add_argument('--lr_patience', default=3, type=int, help='Patience of LR scheduler. '
                                                                   'See documentation of ReduceLROnPlateau.')
    parser.add_argument('--scheduler', default='cyclic', type=str)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--dataset', type=str, default="nyu")
    parser.add_argument('--manual_seed', default=1, type=int,
                        help='Manually set random seed')
    parser.add_argument('--print-freq', '-p', default=1000, type=int,
                        metavar='N', help='print frequency (default: 1000)')
    parser.add_argument('--g_smooth', '-gs', default=False, type=bool,
                        help='Add translational invariance to the attack')
    parser.add_argument('--targeted', '-t', default=False,
                        type=bool, help='Choose if adversarial image is targeted')

    parser.add_argument('--epsilon', default=6.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--k', default=5, type=float)
    parser.add_argument('--move_target', default=0, type=float)

    args = parser.parse_args()
    return args


def create_loader(args):
    if args.dataset == 'kitti':
        kitti_root = Path.db_root_dir(args.dataset)
        if os.path.exists(kitti_root):
            print('kitti dataset "{}" exists!'.format(kitti_root))
        else:
            print('kitti dataset "{}" doesnt existed!'.format(kitti_root))
            exit(-1)

        train_set = kitti_dataloader.KITTIDataset(
            kitti_root, type='train', model=args.model)
        val_set = kitti_dataloader.KITTIDataset(
            kitti_root, type='test', model=args.model)

    elif args.dataset == 'nyu':
        traindir = os.path.join(Path.db_root_dir(args.dataset), 'train')
        if os.path.exists(traindir):
            print('Train dataset "{}" exits!'.format(traindir))
        else:
            print('Train dataset "{}" doesnt existed!'.format(traindir))
            exit(-1)

        valdir = os.path.join(Path.db_root_dir(args.dataset), 'val')
        if os.path.exists(valdir):
            print('Val dataset "{}" exists!'.format(valdir))
        else:
            print('Val dataset "{}" doesnt existed!'.format(valdir))
            exit(-1)

        train_set = nyu_dataloader.NYUDataset(
            traindir, type='train', model=args.model)
        val_set = nyu_dataloader.NYUDataset(
            valdir, type='val', model=args.model)

    elif args.dataset == 'saved_images':
        if not os.path.exists(args.save_image_dir):
            print('Val dataset "{}" doesnt existed!'.format(args.save_image_dir))
            exit(-1)

        train_set = folder_loader.FolderDataset(
            args.save_image_dir, model=args.model)
        val_set = folder_loader.FolderDataset(
            args.save_image_dir, model=args.model)
    else:
        print('no dataset named as ', args.dataset)
        exit(-1)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    if args.dataset == 'kitti':
        if args.save_image_dir is not None:
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        else:
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def get_output_directory(args):
    if args.resume and not args.adv_training:
        return os.path.dirname(args.resume)
    elif args.adv_training:
        save_dir_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)))
        save_dir_root = os.path.join(save_dir_root, 'result',  args.dataset +
                                     "_resnet_" + str(args.resnet_layers) + '_' + args.decoder)
        save_dir = os.path.join(save_dir_root, 'adv_train',
                                str(args.attack) + '_' + str(args.epsilon) + '_' + str(args.lr) + '_' + args.loss)
        return save_dir
    else:
        save_dir_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)))
        save_dir_root = os.path.join(save_dir_root, 'result',  args.dataset +
                                     "_resnet_" + str(args.resnet_layers) + '_' + args.decoder)
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
        return save_dir


def save_checkpoint(state, is_best, epoch, output_directory, every_epoch):
    if every_epoch:
        checkpoint_filename = os.path.join(
            output_directory, 'checkpoint_' + str(epoch) + '.pth.tar')
        torch.save(state, checkpoint_filename)

    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        torch.save(state, best_filename)


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    # H, W, C
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    # H, W, C
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(
        depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(
        depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack(
        [rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
