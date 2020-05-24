# Copyright (C) 2020 Amir Alansary <amiralansary@gmail.com>

import os
import time
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchsummary import summary

from models import NeuralNet, LogisticRegression
from metrics import accuracy
from dataset import HeartDiseaseDataset

# ===================================
best_acc1 = 0
model_names = ['NeuralNet', 'LogisticRegression']
LABELS = ['0', '1', '2', '3', '4']

# ===================================

parser = argparse.ArgumentParser(description='Heart Disease Diagnosis')
parser.add_argument('-s', '--save_dir', metavar='SAVE_DIR',
                    help='path to the save directory', default='models')
parser.add_argument('-t', '--train-file', metavar='TRAIN_FILE', default=None,
                    help='path to the csv file that contain train images')
parser.add_argument('-v', '--valid-file', metavar='VALID_FILE', default=None,
                    help='path to the csv file that contain validation data')
parser.add_argument('-c', '--classes', default=5, type=int,
                    metavar='CLASSES', help='number of classes')
parser.add_argument('-a', '--arch', metavar='ARCH', default='NeuralNet',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: NeuralNet)')
parser.add_argument('-j', '--workers', default=os.cpu_count(),
                    type=int, metavar='N',
                    help='number of data loading workers (default: max)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=500, type=int,
                    metavar='N',
                    help='mini-batch size (default: 100), this is the total')
parser.add_argument('--optim', default='adam', type=str, metavar='OPTIM',
                    help='select optimizer [sgd, adam]',
                    dest='optim')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_scheduler', default=None, type=str, metavar='LR_SCH',
                    help='learning scheduler [reduce, cyclic, cosine]',
                    dest='lr_scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-f', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--suffix', default='', type=str, metavar='SUFFIX',
                    help='add suffix to model save', dest='suffix')
parser.add_argument('--save_results', default='validation_results.csv', type=str,
                    help='Save validation results in a csv file')
parser.add_argument('--ws', '--weighted-sampling', dest='weighted_sampling', action='store_true',
                    help='apply weighted random sampling to balance the classes represented in the mini-batch')
parser.add_argument('--wl', '--weighted-loss', dest='weighted_loss', action='store_true',
                    help='apply weighted loss to balance the classes represented in the mini-batch')


###########################################################################
# main
###########################################################################
def main():
    args = parser.parse_args()

    # seed everything to ensure reproducible results from different runs
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    ###########################################################################
    # Model
    ###########################################################################
    global best_acc1
    # create model
    if args.arch == 'LogisticRegression':
        model = LogisticRegression(input_size=13, n_classes=args.classes)
    elif args.arch == 'NeuralNet':
        model = NeuralNet(input_size=13, hidden_size=[32, 16], n_classes=args.classes) #hidden_size=[64, 32]

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = True
        model = model.cuda(args.gpu)

    # print(model)
    if args.train_file:
        print(30 * '=')
        print(summary(model, input_size=(1, 13),
                      batch_size=args.batch_size, device='cpu'))
        print(30 * '=')

    ###########################################################################
    # save directory
    ###########################################################################
    save_dir = os.path.join(os.getcwd(), args.save_dir)
    save_dir += ('/arch[{}]_optim[{}]_lr[{}]_lrsch[{}]_batch[{}]_'
                 'WeightedSampling[{}]').format(args.arch,
                                                args.optim,
                                                args.lr,
                                                args.lr_scheduler,
                                                args.batch_size,
                                                args.weighted_sampling)
    if args.suffix:
        save_dir += '_{}'.format(args.suffix)
    save_dir = save_dir[:]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ###########################################################################
    # Criterion and optimizer
    ###########################################################################
    # Initialise criterion and optimizer
    if args.gpu is not None:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss()

    # define optimizer
    print("=> using '{}' optimizer".format(args.optim))
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else:  # default is adam
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay,
                                     amsgrad=False)

    ###########################################################################
    # Resume training and load a checkpoint
    ###########################################################################
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ###########################################################################
    # Data Augmentation
    ###########################################################################
    # TODO

    ###########################################################################
    # Learning rate scheduler
    ###########################################################################
    print("=> using '{}' initial learning rate (lr)".format(args.lr))
    # define learning rate scheduler
    scheduler = args.lr_scheduler
    if args.lr_scheduler == 'reduce':
        print("=> using '{}' lr_scheduler".format(args.lr_scheduler))
        # Reduce learning rate when a metric has stopped improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=10)
    elif args.lr_scheduler == 'cyclic':
        print("=> using '{}' lr_scheduler".format(args.lr_scheduler))
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=0.00005,
                                                      max_lr=0.005)
    elif args.lr_scheduler == 'cosine':
        print("=> using '{}' lr_scheduler".format(args.lr_scheduler))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=100,
                                                               eta_min=0,
                                                               last_epoch=-1)

    ###########################################################################
    # load train data
    ###########################################################################
    if args.train_file:
        train_dataset = HeartDiseaseDataset(csv=args.train_file, label_names=LABELS)
        if args.weighted_sampling:
            train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.sample_weights,
                                                                   len(train_dataset),
                                                                   replacement=True)
        else:
            train_sampler = None

        ###########################################################################
        # update criterion
        print('class_sample_count ', train_dataset.class_sample_count)
        print('class_probability ', train_dataset.class_probability)
        print('class_weights ', train_dataset.class_weights)
        print('sample_weights ', train_dataset.sample_weights)

        if args.weighted_loss:
            if args.gpu is not None:
                criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights).cuda(args.gpu)
            else:
                criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    ###########################################################################
    # load validation data
    ###########################################################################
    if args.valid_file:
        valid_dataset = HeartDiseaseDataset(csv=args.valid_file, label_names=LABELS)
        val_loader = torch.utils.data.DataLoader(valid_dataset,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

        if args.evaluate:
            # retrieve correct save path from saved model
            save_dir = os.path.split(args.resume)[0]
            validate(val_loader, model, criterion, save_dir, args)
            return

    ###########################################################################
    # Train the model
    ###########################################################################
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        print_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer,
              scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, save_dir, args)

        # update learning rate based on lr_scheduler
        if args.lr_scheduler == 'reduce':
            scheduler.step(acc1)
        elif args.lr_scheduler == 'cosine':
            scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)

        print("Saving model [{}]...".format(save_dir))
        save_checkpoint({'epoch': epoch + 1,
                         'arch': args.arch,
                         'state_dict': model.state_dict(),
                         'best_acc1': best_acc1,
                         'optimizer': optimizer.state_dict(),
                         'criterion': criterion, },
                        is_best,
                        save_dir=save_dir)
        print(30 * '=')


###############################################################################
# train
###############################################################################
def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top2],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, batch in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        features, target = batch['features'], batch['target']
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            features = features.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(features)
        # compute loss
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), features.size(0))
        top1.update(acc1[0], features.size(0))
        top2.update(acc2[0], features.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update learning rate
        if args.lr_scheduler == 'cyclic':
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


###############################################################################
# Validation
###############################################################################
def validate(val_loader, model, criterion, save_dir, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')

    progress = ProgressMeter(len(val_loader),
                             [batch_time, losses, top1, top2],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # create dataframe to save results in csv file
    if args.save_results:
        results_df = pd.DataFrame(columns=['target', 'predict', 'predict_top2'])

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            # print('batch idx{}, batch len {}'.format(i, len(batch)))
            # get the inputs; data is a list of [inputs, labels]
            features, target = batch['features'], batch['target']

            if args.gpu is not None:
                features = features.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(features)

            # compute loss
            loss = criterion(output, target)

            # Get top2 predictions
            _, pred = output.topk(2, 1, True, True)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), features.size(0))
            top1.update(acc1[0], features.size(0))
            top2.update(acc2[0], features.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update dataframe with new results
            for b in range(len(batch['target'])):
                results_df = results_df.append(
                    dict(target=batch['target'][b].cpu().numpy(), predict=pred[b, 0].cpu().numpy(),
                         predict_top2=pred[b].cpu().numpy()),
                    ignore_index=True)

            if i % args.print_freq == 0:
                progress.display(i)

        # print('params values')
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'.format(top1=top1, top2=top2))

    if args.save_results:
        # Save validation results
        results_file = os.path.join(save_dir, args.save_results)
        results_df.to_csv(results_file, index=False)

    return top1.avg


###############################################################################
# Helper functions (from pytorch examples)
###############################################################################
def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if is_best:
        filename = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        filename = os.path.join(save_dir, filename)

    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        print("Epoch: [{}] Current learning rate (lr) = {}".format(
            epoch, param_group['lr']))


###############################################################################
###############################################################################
if __name__ == '__main__':
    main()
