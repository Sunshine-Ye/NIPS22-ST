# import matplotlib.pyplot as plt
import os
import shutil
import time
import importlib
import argparse


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.mytransforms as mytransforms
from tqdm import tqdm
import numpy as np
import random
import copy
import re
from utils.config import FLAGS
from utils.setlogger import get_logger




parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=str, default='2')
parser.add_argument('--exp', type=str, default='temp')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
exp_root = args.exp

def get_net_device(net):
    return net.parameters().__next__().device

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def smoothing(x, k):
    x_smooth = []
    for i, xx in enumerate(x):
        if i == 0:
            x_smooth.append(xx)
            continue
        x_smooth.append(k * x[i-1] + (1 - k) * xx)
    return x_smooth

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    #
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# def draw_train_and_val_loss():
#     loss_file = exp_root + '/train.log'
#     epochs, train_losses, eval_losses, accs = [], [], [], []
#     with open(loss_file, 'r') as myfile:
#         cnt = 0
#         while(1):
#             line = myfile.readline()
#             if cnt == 0:
#                 cnt += 1
#                 continue
#             if not line:
#                 break
#             line = line.split(',')
#             epochs.append(int(line[0]))
#             train_losses.append(float(line[1]))
#             eval_losses.append(float(line[2]))
#             accs.append(float(line[3]))
#     # plot losses
#     plt.plot(epochs, train_losses, color='b', label='train_losses')
#     plt.plot(epochs, eval_losses, color='r', label='eval_losses')
#     plt.legend()
#     plt.xlabel('Epochs')
#     plt.ylabel('Losses')
#     plt.savefig(exp_root + '/train_val_loss.jpg')
#     print('ok')

def set_running_statistics(model, data_loader):
    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model.module)
    # forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()
            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )
                return lambda_forward
            m.forward = new_forward(m, bn_mean[name], bn_var[name])
    cnt = 0
    with torch.no_grad():
        # DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        forward_model.train()
        for images, labels in data_loader:
            images = images.cuda()
            forward_model(images)
            cnt += images.size(0)
            if cnt >= 4000:
                break
        forward_model.eval()
        # DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if (name in bn_mean and bn_mean[name].count > 0) or (name[7:] in bn_mean and bn_mean[name[7:]].count > 0):
            if name[7:] in bn_mean and bn_mean[name[7:]].count > 0:
                name = name[7:]
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    end = time.time()
    for i, (input, target) in tqdm(enumerate(val_loader)):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return top1.avg, top5.avg, losses.avg

def test_model():
    # checkpoint path
    # ckp_path = exp_root + '/model_best.pth.tar'
    # ckp_path = args.exp + 'model_best.pth.tar'
    ckp_path = exp_root + '/checkpoint.pth.tar'
    ckp = torch.load(ckp_path)
    criterion = nn.CrossEntropyLoss().cuda()
    traindir = os.path.join(FLAGS.dataset_dir, 'train')
    valdir = os.path.join(FLAGS.dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = mytransforms.Lighting(alphastd=0.1)
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            jittering,
            lighting,
            transforms.ToTensor(),
            normalize,
    ])
    # sub_train_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transform
        )

    # sub_train_dataset = datasets.ImageFolder(
    #     valdir,
    #     transform=train_transform
    #     )

    train_sampler = None

    # sub_train_loader = torch.utils.data.DataLoader(
    #     sub_train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
    #     num_workers=FLAGS.workers, pin_memory=True, sampler=train_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
        num_workers=FLAGS.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=FLAGS.workers, pin_memory=True)

    numberofclass = FLAGS.num_classes

    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(depth=FLAGS.depth, num_classes=numberofclass)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(ckp['state_dict'])
    max_width = FLAGS.max_width
    model.apply(lambda m: setattr(m, 'width_mult', max_width))
    # set_running_statistics(model, train_loader)
    set_running_statistics(model, train_loader)
    top1, top5, losses = validate(val_loader, model, criterion)
    print('top1={}, top5={}, loss={}'.format(top1, top5, losses))

def get_epoch_acc_from_log(exp):
    log_path = exp + '/train.log'
    result = {}
    with open(log_path, 'r') as myfile:
        while(1):
            line = myfile.readline()
            # if in the end
            if not line:
                break
            # if search result
            if not re.search('\*', line):
                continue
            # if train
            if re.search('Train Loss', line):
                # log train data
                epoch = int(re.search('\d+', re.search('Epoch: \[\d+/\d+\]', line).group()).group())
                train_loss = float(re.search('\d+.\d+', re.search('Train Loss \d+.\d+', line).group()).group())
                kl_loss = float(re.search('\d+.\d+', re.search('KL_loss \d+.\d+', line).group()).group())
                train_acc = float(re.search('\d+.\d+', re.search('Top 1-acc \d+.\d+', line).group()).group())
                if not epoch in result.keys():
                    result[epoch] = {'train_loss': train_loss, 'train_acc':train_acc, 'kl_loss':kl_loss}
                else:
                    result[epoch]['train_loss'] = train_loss
                    result[epoch]['train_acc'] = train_acc
                    result[epoch]['kl_loss'] = kl_loss
            else:
                # log val data
                epoch = int(re.search('\d+', re.search('Epoch: \[\d+/\d+\]', line).group()).group())
                val_acc = float(re.search('\d+.\d+', re.search('Top 1-acc \d+.\d+', line).group()).group())
                val_loss = float(re.search('\d+.\d+', re.search('Test Loss \d+.\d+', line).group()).group())
                if not epoch in result.keys():
                    result[epoch] = {'val_loss': val_loss, 'val_acc':val_acc}
                else:
                    result[epoch]['val_loss'] = val_loss
                    result[epoch]['val_acc'] = val_acc
    return result

def draw_train_loss_and_val_acc():
    import matplotlib.pyplot as plt
    val_acc, train_loss, epoch = [], [], []
    exps = args.exp.split(',')
    for exp in exps:
        result = get_epoch_acc_from_log(exp)
        temp_val_acc, temp_train_loss, temp_epoch = [], [], []
        for key in result.keys():
            temp_val_acc.append(result[key]['val_acc'])
            temp_train_loss.append(result[key]['train_loss'])
            temp_epoch.append(key)
        val_acc.append(temp_val_acc)
        train_loss.append(temp_train_loss)
        epoch.append(temp_epoch)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    figs = []
    # draw val_acc
    for i, exp in enumerate(exps):
        figs.append(ax1.plot(epoch[i], smoothing(val_acc[i], 0), label=exp + '_acc'))
    ax2 = ax1.twinx()
    # draw train_loss
    for i, exp in enumerate(exps):
        figs.append(ax2.plot(epoch[i], smoothing(train_loss[i], 0), label=exp + '_train_loss'))
    fig_sum = 0
    for fg in figs:
        try:
            fig_sum += fg
        except:
            fig_sum = fg
    ax1.legend(fig_sum, [l.get_label() for l in fig_sum])
    plt.savefig('train_and_acc.png')
    print('ok')

def test_all_model():
    # checkpoint path
    # ckp_path = exp_root + '/model_best.pth.tar'
    # ckp_path = args.exp + 'model_best.pth.tar'
    ckp_path = exp_root + '/checkpoint.pth.tar'
    ckp = torch.load(ckp_path)
    criterion = nn.CrossEntropyLoss().cuda()
    traindir = os.path.join(FLAGS.dataset_dir, 'train')
    valdir = os.path.join(FLAGS.dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = mytransforms.Lighting(alphastd=0.1)
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            jittering,
            lighting,
            transforms.ToTensor(),
            normalize,
    ])
    # sub_train_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transform
        )

    # sub_train_dataset = datasets.ImageFolder(
    #     valdir,
    #     transform=train_transform
    #     )

    train_sampler = None

    # sub_train_loader = torch.utils.data.DataLoader(
    #     sub_train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
    #     num_workers=FLAGS.workers, pin_memory=True, sampler=train_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
        num_workers=FLAGS.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=FLAGS.workers, pin_memory=True)

    numberofclass = FLAGS.num_classes

    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(depth=FLAGS.depth, num_classes=numberofclass)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(ckp['state_dict'])
    max_width = FLAGS.max_width
    model.apply(lambda m: setattr(m, 'width_mult', max_width))
    # set_running_statistics(model, train_loader)
    set_running_statistics(model, train_loader)
    top1, top5, losses = validate(val_loader, model, criterion)
    print('top1={}, top5={}, loss={}'.format(top1, top5, losses))


if __name__=='__main__':
    test_model()
    # get_epoch_acc_from_log()
    # draw_train_loss_and_val_acc()