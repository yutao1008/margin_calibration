"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim


def get_optimizer(args, net):
    """
    Decide Optimizer (SGD or AdamW)
    """
    param_groups = net.parameters()
    if args.sgd:
        optimizer = optim.SGD(param_groups,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum,
                                nesterov=False)
    elif args.adamw:
        optimizer = optim.AdamW(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'poly':
        lambda1 = lambda iteration: math.pow(1 - iteration / args.max_iter, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


def load_weights(net, optimizer, scheduler, snapshot_file, restore_optimizer=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer, scheduler, epoch, mean_iou = restore_snapshot(net, optimizer, scheduler, snapshot_file,
            restore_optimizer)
    return epoch, mean_iou


def restore_snapshot(net, optimizer, scheduler, snapshot, restore_optimizer):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot)
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer:
        scheduler.load_state_dict(checkpoint['scheduler'])
    net.load_state_dict(checkpoint['state_dict'])
    return net, optimizer, scheduler, checkpoint['epoch'], checkpoint['mean_iou']
