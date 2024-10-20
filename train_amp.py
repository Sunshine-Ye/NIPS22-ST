import math
import copy
import argparse
import os
import shutil
import time
import importlib
import aug_lib
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
import numpy as np
import random
from utils.config import FLAGS
from utils.setlogger import get_logger

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=50)
# parser.add_argument('--GPU', type=str, default='1, 2, 3')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--exp', type=str, default='ut_1_1_50')
parser.add_argument('--kl_coef', type=float, default=1)
parser.add_argument('--main_coef', type=float, default=1.0)
parser.add_argument('--UT', action='store_true', default=True)
parser.add_argument('--kl_cos', action='store_true', default=False)
parser.add_argument('--kl_rcos', action='store_true', default=False)
parser.add_argument('--val_tf_bn_reset', action='store_true', default=False)
# parser.add_argument('--yaml_file', type=str, default='configs/resnet50_randwidth.yml')
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=0.0001)
# multiple subnet
parser.add_argument('--multi_Snet', type=int, default=1)
# directional KD
parser.add_argument('--norm_kd', action='store_true', default=False)
parser.add_argument('--norm_kd_test', action='store_true', default=False)
parser.add_argument('--amplitude', type=float, default=2)
parser.add_argument('--Temp', type=float, default=1)
# directional KD ensemble
parser.add_argument('--direct_KD_esm', action='store_true', default=False)
parser.add_argument('--direct_KD_ema', type=float, default=0.5)
# subnet data transform
parser.add_argument('--Snet_Dtrans', action='store_true', default=False)
parser.add_argument('--Dtrans', type=str, default='Dtrans6')
# min lr
parser.add_argument('--min_lr', action='store_true', default=False)
parser.add_argument('--min_lr_coef', type=float, default=0.0001)
# more tricks
parser.add_argument('--val_resize_size', type=int, default=256)  # common: val_resize_256+val_crop_224
parser.add_argument('--val_crop_size', type=int, default=224)
parser.add_argument('--train_crop_size', type=int, default=224)
parser.add_argument('--mini_size', type=float, default=0.43)  # int(args.train_crop_size*args.mini_size)=int(224*0.43)=96
parser.add_argument('--jitter_coef', type=float, default=0.4)
parser.add_argument('--light_coef', type=float, default=0.1)
parser.add_argument('--train_trans', type=str, default='default')
parser.add_argument('--mix', type=str, default='default')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--max_size', type=float, default=1.0)  #TODO： check！
parser.add_argument('--sample_space', type=str, default='default')
args = parser.parse_args()
# FLAGS = Config(args.yaml_file)
# os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# saved_path = FLAGS.log_dir
saved_path = args.exp
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, 'train.log'))
best_acc1 = 0
best_acc5 = 0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
jittering = transforms.ColorJitter(brightness=args.jitter_coef, contrast=args.jitter_coef, saturation=args.jitter_coef)
lighting = mytransforms.Lighting(alphastd=args.light_coef)

def main():
    global best_acc1, best_acc5

    traindir = os.path.join('/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet', 'train')
    valdir = os.path.join('/mnt/petrelfs/yepeng/datasets/classification/imagenet', 'val')
    if args.train_trans == 'default':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                transforms.ToTensor(),
                normalize,
        ])
    elif args.train_trans == 'with_TA_torch':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                normalize,
        ])
    elif args.train_trans == 'with_TA_torch1':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                transforms.TrivialAugmentWide(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                normalize,
        ])
    elif args.train_trans == 'with_TA':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                aug_lib.TrivialAugment(),
                transforms.ToTensor(),
                normalize,
        ])
    elif args.train_trans == 'with_RA_torch':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                transforms.RandAugment(num_ops=1, magnitude=9),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                normalize,
        ])
    elif args.train_trans == 'with_RA':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                aug_lib.RandAugment(n=1, m=9),  # M/N=9/1(), M/N=8/2(79.952), M/N=7/3(79.72); timm/MSTD
                transforms.ToTensor(),
                normalize,
        ])
    elif args.train_trans == 'with_UA':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                aug_lib.UniAugment(),
                transforms.ToTensor(),
                normalize,
        ])
    elif args.train_trans == 'with_AA':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                # 这里policy属于torchvision.transforms.autoaugment.AutoAugmentPolicy，
                # 对于ImageNet就是 AutoAugmentPolicy.IMAGENET
                # 此时aa_policy = autoaugment.AutoAugmentPolicy('imagenet')
                transforms.AutoAugment(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                normalize,
        ])
    elif args.train_trans == 'with_RE':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(args.train_crop_size),
                transforms.RandomHorizontalFlip(),
                jittering,
                lighting,
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                normalize,
                # scale是指相对于原图的擦除面积范围
                # ratio是指擦除区域的宽高比
                # value是指擦除区域的值，如果是int，也可以是tuple（RGB3个通道值），或者是str，需为'random'，表示随机生成
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ])
    else:
        raise NotImplementedError

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transform
        )
    if args.mix == 'default':
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    elif args.mix == 'cutmix':
        mixup_transform = RandomCutmix(1000, alpha=1.0, p=1.0)
        collate_fn = lambda batch: mixup_transform(*torch.utils.data.dataloader.default_collate(batch))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    elif args.mix == 'mixup':
        mixup_transform = RandomMixup(1000, alpha=args.mixup_alpha, p=1.0)
        collate_fn = lambda batch: mixup_transform(*torch.utils.data.dataloader.default_collate(batch))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    elif args.mix == 'cutmix-torch':
        import torch_mixup
        mixup_transforms = [torch_mixup.RandomCutmix(1000, p=1.0, alpha=1.0)]
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*torch.utils.data.dataloader.default_collate(batch))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    elif args.mix == 'mixup-torch':
        import torch_mixup
        mixup_transforms = [torch_mixup.RandomMixup(1000, p=1.0, alpha=args.mixup_alpha)]
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*torch.utils.data.dataloader.default_collate(batch))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    elif args.mix == 'mixup-cutmix-torch':
        import torch_mixup
        mixup_transforms = [torch_mixup.RandomMixup(1000, p=1.0, alpha=args.mixup_alpha), torch_mixup.RandomCutmix(1000, p=1.0, alpha=1.0)]
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*torch.utils.data.dataloader.default_collate(batch))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    elif args.mix == 'mixup-cutmix-timm':
        from timm_mixup import Mixup, FastCollateMixup
        mixup_transform = Mixup(mixup_alpha=args.mixup_alpha, cutmix_alpha=1.0, prob=1.0,
                                switch_prob=0.5, label_smoothing=0.0, num_classes=1000)
        collate_fn = lambda batch: mixup_transform(*torch.utils.data.dataloader.default_collate(batch))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn, drop_last=True)
    else:
        raise NotImplementedError

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(args.val_resize_size),
            transforms.CenterCrop(args.val_crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size//2, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if args.val_tf_bn_reset:  # note: whether using val tf for bn reset
        sub_train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size // 2, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    numberofclass = FLAGS.num_classes

    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model_amp(res_depth=args.depth, num_classes=numberofclass)
    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=FLAGS.momentum,
                                weight_decay=args.weight_decay, nesterov=FLAGS.nesterov)
    lr_scheduler = get_lr_scheduler(optimizer, train_loader)

    if FLAGS.test_only:
        ckpt = torch.load(FLAGS.pretrained)
        model.load_state_dict(ckpt['model'], strict=True)
        print('Load pretrained weights from ', FLAGS.pretrained)
        acc1, acc5, _ = validate(val_loader, model, criterion, 0)
        print('Top-1 and 5 accuracy:', acc1, acc5)
        return

    start_epoch = 0
    if args.resume:
        print('Resume chkp ... ')
        # ckpt = torch.load(os.path.join(args.exp, 'checkpoint.pth.tar'))
        ckpt = torch.load(os.path.join(args.exp, 'model_best.pth.tar'))
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(optimizer.param_groups[0]['lr'])
        start_epoch = ckpt['epoch'] + 1
        lr_scheduler = get_lr_scheduler(optimizer, train_loader, last_epoch=start_epoch)
        best_acc1 = ckpt['best_acc1']
        best_acc5 = ckpt['best_acc5']
        print('best_acc1', best_acc1)
        print('best_acc5', best_acc5)
        print('Load weights from epoch', start_epoch)
        acc1, acc5, val_loss = validate(val_loader, model, criterion, args.epoch)  # used for inference resize tuning

    for epoch in range(start_epoch, args.epoch):

        # train for one epoch
        scaler = torch.cuda.amp.GradScaler()
        train_loss = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, scaler)

        # evaluate on validation set
        set_running_statistics(model, train_loader)  # note: re-set bn
        acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc5 = acc5
            with open(args.exp + '/best_acc.txt', 'w') as myfile:  # note： log the best acc
                myfile.write('best epoch:{}, acc is {}'.format(epoch, best_acc1))

        print('Current best accuracy (top-1 and 5 accuracy):', best_acc1, best_acc5)
        save_checkpoint({
            'epoch': epoch,
            # 'arch': FLAGS.net_type,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print('Best accuracy (top-1 and 5 accuracy):', best_acc1, best_acc5)

# note：re-set bn statistics
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
        forward_model.train()  # TODO：check the influence of dropout
        for images, labels in data_loader:
            images = images.cuda()
            forward_model(images)
            cnt += images.size(0)
            if cnt >= 2000:
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


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_kl = AverageMeter()  # note：record kl loss

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        current_LR = get_learning_rate(optimizer)[0]
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        # first do max_width and max_resolution
        max_width = FLAGS.max_width
        model.apply(lambda m: setattr(m, 'width_mult', max_width))
        if args.max_size != 1.0:
            aux_size = random.randint(args.train_crop_size, int(args.train_crop_size * args.max_size), )
            Snet_Dtrans = transforms.Compose([
                transforms.Resize((aux_size, aux_size)),
                normalize,
            ])
            input = Snet_Dtrans(input)
        with torch.cuda.amp.autocast():
            max_output = model(input)
            if args.norm_kd:  # remove the influence of amplitude of teacher logits
                # max_output = F.layer_norm(max_output, torch.Size((1000,)), None, None, 1e-7)*args.amplitude
                # max_output = F.layer_norm(max_output, torch.Size((1000,)), None, None, 1e-12) * args.amplitude
                max_output = F.normalize(max_output, p=2.0, dim=-1, eps=1e-7)
            loss = torch.mean(criterion(max_output, target))
        max_output_detach = max_output.detach()
        # (args.main_coef * loss).backward()  # note：main loss backward
        scaler.scale(args.main_coef * loss).backward()
        # import pdb
        # pdb.set_trace()
        # logger.info(args.multi_Snet)
        for _ in range(args.multi_Snet):
            # note：do UT sample and update
            model.module.depth = sample(model.module.res_depth)
            # logger.info(model.module.depth)
            if args.Snet_Dtrans:
                # print(input.size())
                # Snet_Dtrans = transforms.Compose([  # original
                #     # transforms.ToPILImage(),  # TODO:check
                #     transforms.RandomResizedCrop(224),
                #     transforms.RandomHorizontalFlip(),
                #     jittering,
                #     # lighting,
                #     # transforms.ToTensor(),  # TODO:check
                #     normalize,  # TODO:check
                # ])
                # Snet_Dtrans = transforms.Compose([  # v1
                #     transforms.RandomResizedCrop(224),
                #     jittering,
                #     normalize,
                # ])
                # Snet_Dtrans = transforms.Compose([  # v2
                #     transforms.RandomResizedCrop(224),
                #     normalize,
                # ])
                # Snet_Dtrans = transforms.Compose([  # v3
                #     transforms.RandomResizedCrop(224),
                # ])
                # input = Snet_Dtrans(input)
                if args.Dtrans == 'Dtrans4':
                    idx = random.randint(0, len(FLAGS.resos) - 1)  # v4  FLAGS.resos=[224, 192, 160, 128]
                    input = F.interpolate(input, (FLAGS.resos[idx], FLAGS.resos[idx]), mode='bilinear',
                                          align_corners=True)
                elif args.Dtrans == 'Dtrans5':
                    idx = random.randint(0, len(FLAGS.resos) - 1)  # v5  FLAGS.resos=[224, 192, 160, 128]
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((FLAGS.resos[idx], FLAGS.resos[idx])),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans6':
                    aux_size = random.randint(128, 224)  # v6
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans7':
                    # aux_size = random.randint(96, 224)  # v7
                    aux_size = random.randint(int(args.train_crop_size*args.mini_size), args.train_crop_size)
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans8':
                    aux_size = random.randint(64, 224)  # v8
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans9':
                    aux_size = random.randint(96, 256)  # v9
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans10':
                    aux_size = random.randint(32, 224)  # v10
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans11':
                    aux_size = random.randint(96, 240)  # v11
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans12':
                    aux_size = random.randint(80, 224)  # v12
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans13':
                    aux_size = random.randint(96, 208)  # v13
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                elif args.Dtrans == 'Dtrans14':
                    aux_size = random.randint(96, 272)  # v14
                    Snet_Dtrans = transforms.Compose([
                        transforms.Resize((aux_size, aux_size)),
                        normalize,
                    ])
                    input = Snet_Dtrans(input)
                else:
                    NotImplementedError
                # logger.info(input.size())
            with torch.cuda.amp.autocast():
                if args.UT:
                    output = model(input)
                else:
                    with torch.no_grad():
                        output = model(input)
                if args.norm_kd:  # remove the influence of amplitude of student logits
                    # output = F.layer_norm(output, torch.Size((1000,)), None, None, 1e-7)  # *args.amplitude  #TODO: asymmetric?
                    # output = F.layer_norm(output, torch.Size((1000,)), None, None, 1e-12)
                    output = F.normalize(output, p=2.0, dim=-1, eps=1e-7)
                # kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/args.Temp, dim=1),
                #                             F.softmax(max_output_detach/args.Temp, dim=1))*args.Temp*args.Temp
                kl_loss = F.kl_div(
                    F.log_softmax(output/args.Temp, dim=1),
                    # We provide the teacher's targets in log probability because we use log_target=True
                    # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                    # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                    F.log_softmax(max_output_detach/args.Temp, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (args.Temp*args.Temp) / output.numel()
                # We divide by output.numel() to have the legacy PyTorch behavior.
                # But we also experiments output_kd.size(0)
                # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details

            if args.direct_KD_esm:
                max_output_detach = max_output_detach*(1-args.direct_KD_ema)+\
                                F.layer_norm(output.detach(), torch.Size((1000,)), None, None, 1e-7)*args.direct_KD_ema
            if args.UT:
                if args.kl_cos:
                    cos_coef = args.kl_coef * 0.5 * (1 - math.cos(math.pi * epoch / args.epoch))
                    (cos_coef * kl_loss).backward()
                elif args.kl_rcos:
                    rcos_coef = args.kl_coef - args.kl_coef * 0.5 * (1 - math.cos(math.pi * epoch / args.epoch))
                    (rcos_coef * kl_loss).backward()
                else:
                    # (args.kl_coef * kl_loss).backward()
                    scaler.scale(args.kl_coef * kl_loss).backward()
        model.module.depth = model.module.block_setting_dict[model.module.res_depth]
        # print(model.module.depth)

        # measure accuracy and record loss
        # import pdb; pdb.set_trace()
        if args.mix == 'default':
            acc1, acc5 = accuracy(max_output.data, target, topk=(1, 5))
        elif args.mix == 'cutmix' or args.mix == 'mixup':
            acc1, acc5 = accuracy(max_output.data, torch.argmax(target, -1, keepdim=False), topk=(1, 5))
        else:
            acc1, acc5 = accuracy(max_output.data, torch.argmax(target, -1, keepdim=False), topk=(1, 5))
            # raise NotImplementedError
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # # do other widths and resolution
        # min_width = FLAGS.min_width
        # width_mult_list = [min_width]
        # sampled_width = list(np.random.uniform(min_width, max_width, FLAGS.num_subnet-1))
        # width_mult_list.extend(sampled_width)
        # for width_mult in sorted(width_mult_list, reverse=True):
        #     model.apply(
        #         lambda m: setattr(m, 'width_mult', width_mult))
        #     idx = random.randint(0, len(FLAGS.resos) - 1)
        #     output = model(F.interpolate(input, (FLAGS.resos[idx], FLAGS.resos[idx]), mode='bilinear', align_corners=True))
        #     loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1),
        #                                                      F.softmax(max_output_detach, dim=1))
        #     loss.backward()

        losses_kl.update(kl_loss.item(), input.size(0))  # note：record kl loss
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        if args.min_lr:
            for param_group in optimizer.param_groups:
                if param_group['lr'] <= args.lr*args.min_lr_coef:
                    param_group['lr'] = args.lr*args.min_lr_coef

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % FLAGS.print_freq == 0:
            logger.info('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_kl {losses_kl.val:.4f} ({losses_kl.avg:.4f})\t'  # note：record kl loss
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epoch, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, losses_kl=losses_kl, loss=losses, top1=top1, top5=top5))

    logger.info('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Train Loss {loss.avg:.3f}\t KL_loss {loss_kl.avg:.3f}'.format(
        epoch, args.epoch, top1=top1, top5=top5, loss=losses, loss_kl=losses_kl))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        if args.norm_kd_test:  # remove the influence of amplitude of teacher logits
            output = F.layer_norm(output, torch.Size((1000,)), None, None, 1e-7)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % FLAGS.print_freq == 0:
            logger.info('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    logger.info('* Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epoch, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = args.exp
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + '/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + '/model_best.pth.tar')


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


def get_lr_scheduler(optimizer, trainloader, last_epoch=-1):
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma, last_epoch=last_epoch)
    elif FLAGS.lr_scheduler == 'cosine':
        if last_epoch != -1:
            last_epoch = last_epoch*len(trainloader)-1
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch*len(trainloader), last_epoch=last_epoch)
    else:
        raise NotImplementedError('LR scheduler not implemented.')
    return lr_scheduler


# note: sample depth-level sub-networks
def sample(res_depth=50):
    if res_depth == 50:
        depth_st = []
        if args.sample_space == 'default':
            depth_candi_v6 = [list(range(3)), list(range(4)), list(range(6)), list(range(3))]
        elif args.sample_space == 'space1':
            depth_candi_v6 = [list(range(2)), list(range(3)), list(range(5)), list(range(2))]
        elif args.sample_space == 'space2':
            depth_candi_v6 = [list(range(2)), list(range(3)), list(range(6)), list(range(3))]
        elif args.sample_space == 'space3':
            depth_candi_v6 = [list(range(1)), list(range(2)), list(range(4)), list(range(1))]
        elif args.sample_space == 'space4':
            depth_candi_v6 = [list(range(1)), list(range(2)), list(range(5)), list(range(2))]
        elif args.sample_space == 'space5':
            depth_candi_v6 = [list(range(1)), list(range(1)), list(range(3)), list(range(1))]
        elif args.sample_space == 'space6':
            depth_candi_v6 = [list(range(1)), list(range(2)), list(range(4)), list(range(2))]
        else:
            raise NotImplementedError
        resnet50_depth = [3, 4, 6, 3]
        # depth_candi_v1 = [[0], [0, 1, 2], [0, 1, 2, 3], [0, 1]]
        # import pdb; pdb.set_trace()
        target_net = resnet50_depth
        depth_candi_ad = depth_candi_v6
        for depth_candi_i, depth in zip(depth_candi_ad, target_net):
            depth_st.append(depth - np.random.choice(depth_candi_i, 1)[0])
    elif res_depth == 101:
        depth_st = []
        depth_candi_v6 = [list(range(3)), list(range(4)), list(range(23)), list(range(3))]
        resnet50_depth = [3, 4, 23, 3]
        target_net = resnet50_depth
        depth_candi_ad = depth_candi_v6
        for depth_candi_i, depth in zip(depth_candi_ad, target_net):
            depth_st.append(depth - np.random.choice(depth_candi_i, 1)[0])
    elif res_depth == 152:
        depth_st = []
        depth_candi_v6 = [list(range(3)), list(range(8)), list(range(36)), list(range(3))]
        resnet50_depth = [3, 8, 36, 3]
        target_net = resnet50_depth
        depth_candi_ad = depth_candi_v6
        for depth_candi_i, depth in zip(depth_candi_ad, target_net):
            depth_st.append(depth - np.random.choice(depth_candi_i, 1)[0])
    return depth_st

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     if args.dataset.startswith('cifar'):
#         lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
#     elif args.dataset == ('imagenet'):
#         if args.epochs == 300:
#             lr = args.lr * (0.1 ** (epoch // 75))
#         else:
#             lr = args.lr * (0.1 ** (epoch // 30))
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    # print(lr)
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

from typing import Tuple
from torch import Tensor
# from https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0. # beta分布超参数
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        # 建立one-hot标签
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        # 判断是否进行mixup
        if torch.rand(1).item() >= self.p:
            return batch, target

        # 这里将batch数据平移一个单位，产生mixup的图像对，这意味着每个图像与相邻的下一个图像进行mixup
        # timm实现是通过flip来做的，这意味着第一个图像和最后一个图像进行mixup
        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # 随机生成组合系数
        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)  # 得到mixup后的图像

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)  # 得到mixup后的标签

        return batch, target

class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        # W, H = F.get_image_size(batch)
        # W, H = F._get_image_size(batch)
        W, H = batch.shape[-2], batch.shape[-1]
        # logger.info('* W/H: [{0}/{1}]'.format(W, H))
        # import pdb; pdb.set_trace()

        # 确定patch的起点
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        # 确定patch的w和h（其实是一半大小）
        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        # 越界处理
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        # 由于越界处理， λ可能发生改变，所以要重新计算
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

if __name__ == '__main__':
    main()
