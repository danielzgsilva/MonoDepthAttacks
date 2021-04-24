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

from attacks.MIFGSM import MIFGSM
from attacks.pgd import PGD


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU


def main():
    global args, best_result, output_directory

    print(torch.__version__)

    # set random seed
    torch.manual_seed(args.manual_seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        print(torch.version.cuda)
        print(torch.cuda.device_count())
        print(torch.cuda.is_available())
        print()

        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use GPU ", torch.cuda.current_device())

    train_loader, val_loader = utils.create_loader(args)

    # load model
    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
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
    else:
        print("=> creating Model from scratch")
        model = FCRN.ResNet(layers=args.resnet_layers, output_size=train_loader.dataset.output_size)
        start_epoch = 0

        # different modules have different learning rate
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            assert(False, "{} optim not supported".format(args.optim))

        # You can use DataParallel() whether you use Multi-GPUs or not
        model = nn.DataParallel(model).cuda()

    attacker = None
    if args.adv_training:
        start_epoch = 0
        if not args.attack:
            assert(False, "You must supply an attack for adversarial training")

        if args.attack == 'mifgsm':
            attacker = MIFGSM(model, "cuda:0", args.loss,
                              eps=mifgsm_params['eps'],
                              steps=mifgsm_params['steps'],
                              decay=mifgsm_params['decay'],
                              alpha=mifgsm_params['alpha'],
                              TI=mifgsm_params['TI'],
                              k_=mifgsm_params['k'],
                              targeted=args.targeted,
                              test=args.model)
        elif args.attack == 'pgd':
            attacker = PGD(model, "cuda:0", args.loss,
                           norm=pgd_params['norm'],
                           eps=pgd_params['eps'],
                           alpha=pgd_params['alpha'],
                           iters=pgd_params['iterations'],
                           TI=pgd_params['TI'],
                           k_=mifgsm_params['k'],
                           test=args.model)
        else:
            assert(False, "{} attack not supported".format(args.attack))

        print('performing adversarial training with {} attack and {} loss'.format(args.attack, args.loss))
    else:
        print('performing standard training with {} loss'.format(args.loss))

    # when training, use reduceLROnPlateau to reduce learning rate
    if args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience)
    elif args.scheduler == 'cyclic':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr * 100)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, eta_min=0.000001, T_mult=2)
    else:
        scheduler = None

    # loss function
    if args.loss == 'l1':
        criterion = criteria.MaskedL1Loss()
    elif args.loss == 'l2':
        criterion = criteria.MaskedMSELoss()
    elif args.loss == 'berhu':
        criterion = criteria.berHuLoss()
    else:
        assert(False, '{} loss not supported'.format(args.loss))

    # create directory path
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)

    # create log
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)

    for epoch in range(start_epoch, args.epochs):

        # remember change of the learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])

        train(train_loader, model, criterion, optimizer, epoch, attacker)  # train for one epoch
        result, img_merge = validate(val_loader, model, epoch)  # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}, rmse={:.3f}, rml={:.3f}, log10={:.3f}, d1={:.3f}, d2={:.3f}, dd31={:.3f}, "
                    "t_gpu={:.4f}".
                        format(epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                               result.delta3,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        # save checkpoint for each epoch
        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

        # when rml doesn't fall, reduce learning rate
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(result.rmse)
            elif args.scheduler == 'cyclic':
                scheduler.step()
            elif args.scheduler == 'cosine':
                scheduler.step()


# train
def train(train_loader, model, criterion, optimizer, epoch, attacker):
    average_meter = AverageMeter()
    model.train()  # switch to train mode
    end = time.time()

    batch_num = len(train_loader)

    for i, (input, target) in enumerate(train_loader):
        # itr_count += 1
        input, target = input.cuda(), target.cuda()

        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        # get adversary for adversarial training
        if attacker is not None:
            input = attacker(input, target)

        pred = model(input)

        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))
            current_step = epoch * batch_num + i

    avg = average_meter.average()


# validation
def validate(val_loader, model, epoch):
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
            pred = model(input)

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
    max_perturb = 6.0
    iterations = 10
    alpha = 1.0
    TI = False
    k = 5

    mifgsm_params = {'eps': max_perturb, 'steps': iterations, 'decay': 1.0, 'alpha': alpha, 'TI': TI, 'k': k}
    pgd_params = {'norm': 'inf', 'eps': max_perturb, 'alpha': alpha, 'iterations': iterations, 'TI': TI, 'k': k}

    args = utils.parse_command()
    print(args)

    best_result = Result()
    best_result.set_to_worst()

    main()
