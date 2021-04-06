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
import torch.nn.functional as F
from torch.optim import lr_scheduler

from network import FCRN
from utils import criteria, utils
from utils.metrics import AverageMeter, Result

from AdaBins.models import UnetAdaptiveBins
from AdaBins import model_io

from DPT.dpt.models import DPTDepthModel

from attacks.MIFGSM import MIFGSM

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU


def main():
    global args, best_result, output_directory

    print(torch.__version__)

    # set random seed
    torch.manual_seed(args.manual_seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use GPU ", torch.cuda.current_device())

    _, val_loader = utils.create_loader(args)
    print("Kitti dataloader loaded")
    del _

    if args.resume is not None:
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
    
    elif args.model == "dpt":
        attention_hooks = True

        model = DPTDepthModel(
            path=args.resume,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=attention_hooks,
        )
        print('model {} loaded'.format(args.resume))
    else:
        assert(False, "{} model not supported".format(args.model))

    torch.cuda.empty_cache()
    model = model.cuda()

    attacker = None
    if args.attack == 'mifgsm':
        print('attacking with {}'.format(args.attack))
        attacker = MIFGSM(model, "cuda:0", args.loss,
                          eps=mifgsm_params['eps'],
                          steps=mifgsm_params['steps'],
                          decay=mifgsm_params['decay'],
                          alpha=mifgsm_params['alpha'],
                          TI=mifgsm_params['TI'],
                          k_=mifgsm_params['k'],
                          test=args.model)
    else:
        print('no attack')

    # create directory path
    if args.eval_output_dir is not None:
        output_directory = args.eval_output_dir
    else:
        output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_results')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    eval_txt = os.path.join(output_directory, 'eval_results_{}_{}_{}.txt'.format(
                                                                        args.model, args.dataset, args.attack))

    # evaluate on validation set
    result, img_merge = validate(val_loader, model, attacker)

    with open(eval_txt, 'w') as txtfile:
        txtfile.write(
            "rmse={:.3f}, rml={:.3f}, log10={:.3f}, d1={:.3f}, d2={:.3f}, dd31={:.3f}, t_gpu={:.4f}".
                format(result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                       result.delta3, result.gpu_time))

    if img_merge is not None:
        img_filename = output_directory + '/eval_results_{}_{}_{}.png'.format(
                                                                    args.model, args.dataset, args.attack)
        utils.save_image(img_merge, img_filename)


# validation
def validate(val_loader, model, attacker):
    average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode

    end = time.time()

    skip = len(val_loader) // 9  # save images every skip iters

    for i, (input, target) in enumerate(val_loader):

        input, target = input.cuda(), target.cuda()
        input_clone = input.clone(); target_clone = target.clone()

        if args.model == 'dpt':
            input_clone = torch.nn.functional.interpolate(
                    input_clone,
                    size=(384, 384),
                    mode="bicubic",
                    align_corners=False)
            target_clone = torch.nn.functional.interpolate(
                    target_clone,
                    size=(384, 384),
                    mode="bicubic",
                    align_corners=False)

        adv_input = get_adversary(input_clone, target_clone, attacker)

        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            if args.model == 'adabins':
                _, pred = model(adv_input)
            else:
                pred = model(adv_input)

        input, target, pred = post_process(input, target, pred, args.model)
        # print('pred {} \n target {}'.format(pred, target))
        # print(input.shape, target.shape, pred.shape)


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
            target = target[0]
            pred = pred[0]
        else:
            rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)

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


def get_adversary(data, target, attacker=None):
    if attacker is not None:
        pert_image = attacker(data, target)
    else:
        pert_image = data

    return pert_image


def post_process(input, target, depth, model=None, bits=1):
    if model == 'adabins':
        # resize target to match adabins output size
        if args.dataset == 'kitti':
            #target = F.interpolate(target, size=(114, 456), mode='bilinear')
            #input = F.interpolate(input, size=(114, 456), mode='bilinear')
            depth = F.interpolate(depth, size=(228, 912), mode='bilinear')
        elif args.dataset == 'nyu':
            #target = F.interpolate(target, size=(480, 640))
            #input = F.interpolate(input, size=(480, 640), mode='bilinear')
            depth = F.interpolate(depth, size=(480, 640), mode='bilinear')
    
    elif model == 'dpt':
        depth_min = torch.min(depth)
        depth_max = torch.max(depth)

        # TODO What am I even doing?
        depth[depth < 0.5] = 0
        depth = (255 - (255 * (depth - depth_min) / (depth_max - depth_min))) / 10
        F.relu(depth, inplace=True)

        depth = torch.nn.functional.interpolate(
                            depth.unsqueeze(1),
                            size=target.shape[2:],
                            mode="bicubic",
                            align_corners=False,)
    else:
        pass

    return input, target, depth


if __name__ == '__main__':
    max_perturb = 5.0
    iterations = 10
    alpha = 1.0
    TI = False
    k = 5

    mifgsm_params = {'eps': max_perturb, 'steps': iterations, 'decay': 1.0, 'alpha': alpha, 'TI': TI, 'k': k}

    args = utils.parse_command()
    print(args)

    best_result = Result()
    best_result.set_to_worst()

    main()
