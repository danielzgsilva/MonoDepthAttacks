# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 15:25
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from datetime import datetime
import shutil
import socket
import time
import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from network import FCRN
from utils import criteria, utils
from utils.metrics import AverageMeter, Result

from AdaBins.models import UnetAdaptiveBins
from AdaBins import model_io


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU


def main():
    global args, best_result, output_directory

    # set random seed
    torch.manual_seed(args.manual_seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use GPU ", torch.cuda.current_device())

    _, val_loader = utils.create_loader(args)
    del _

    assert os.path.isfile(args.resume), \
        "=> no checkpoint found at '{}'".format(args.resume)
    print("=> loading checkpoint '{}'".format(args.resume))

    if args.model == 'resnet':
        checkpoint = torch.load(args.resume)

        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer = checkpoint['optimizer']

        # model_dict = checkpoint['model'].module.state_dict()  # to load the trained model using multi-GPUs
        # model = FCRN.ResNet(output_size=train_loader.dataset.output_size, pretrained=False)
        # model.load_state_dict(model_dict)

        # solve 'out of memory'
        model = checkpoint['model']

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        # clear memory
        del checkpoint
        # del model_dict
        torch.cuda.empty_cache()
    elif args.model == "adabins":
        MIN_DEPTH = 1e-3
        MAX_DEPTH_NYU = 10
        MAX_DEPTH_KITTI = 80
        N_BINS = 256

        if args.dataset == 'kitti':
            model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
        elif args.dataset == 'nyu':
            model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
        else:
            assert (False, "{} dataset not supported".format(args.dataset))

        model, _, _ = model_io.load_checkpoint(args.resume, model)

    else:
        assert(False, "{} model not supported".format(args.model))

    # create directory path
    if args.eval_output_dir is not None:
        output_directory = args.eval_output_dir
    else:
        output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_results')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    eval_txt = os.path.join(output_directory, 'eval_results_{}.txt'.format(args.model))

    result, img_merge = validate(val_loader, model)  # evaluate on validation set

    with open(eval_txt, 'w') as txtfile:
        txtfile.write(
            "rmse={:.3f}, rml={:.3f}, log10={:.3f}, d1={:.3f}, d2={:.3f}, dd31={:.3f}, t_gpu={:.4f}".
                format(result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                       result.delta3, result.gpu_time))

    if img_merge is not None:
        img_filename = output_directory + '/eval_results.png'
        utils.save_image(img_merge, img_filename)

# validation
def validate(val_loader, model):
    average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode

    end = time.time()

    skip = len(val_loader) // 9  # save images every skip iters

    for i, (input, target) in enumerate(val_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            if args.model != 'adabins':
                pred = model(input)
            else:
                _, pred = model(input)

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)

        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        if args.dataset == 'kitti':
            rgb = input[0]
            pred = pred[0]
            target = target[0]
        else:
            rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            # filename = output_directory + '/comparison_' + str(epoch) + '.png'
            # utils.save_image(img_merge, filename)
            pass

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'Rel={average.absrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    return avg, img_merge


if __name__ == '__main__':
    args = utils.parse_command()
    print(args)

    best_result = Result()
    best_result.set_to_worst()

    main()
