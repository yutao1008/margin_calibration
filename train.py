import argparse
import logging
import os
import torch
from network.coplenet import COPLENet
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import optimizer
import time
import numpy as np

parser = argparse.ArgumentParser(description='CopleNet for medical image segmentation')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--dataset', type=str, default='robotic_instrument',
                    help='robotic_instrument, covid19_lesion')
parser.add_argument('--task', type=str, default='parts', help='classification task')

# Loss functions
parser.add_argument('--ce2d_loss', action='store_true', default=False,
                    help='2d-crossentropy loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='using class weighted loss')
parser.add_argument('--dice_loss', action='store_true', default=False,
                    help='using generalized dice loss')
parser.add_argument('--focal_loss', action='store_true', default=False,
                    help='using focal loss')
parser.add_argument('--tversky_loss', action='store_true', default=False,
                    help='using Tversky loss')
parser.add_argument('--lovasz_softmax', action='store_true', default=False,
                    help='lovasz softmax loss')
parser.add_argument('--margin_loss', action='store_true', default=False,
                    help='using margin loss')
parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--adamw', action='store_true', default=False)

parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--max_iter', type=int, default=50000)

# Optimization options
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training and validation')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--date', type=str, default='0726',
                    help='experiment directory date name')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')



args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iou': 0, 'fwavacc': 0}

torch.backends.cudnn.benchmark = True

def main():
    """
    Main Function
    """
    prep_experiment(args, parser)
    if args.dataset=='robotic_instrument':
        from datasets.robotic_instrument import get_dataloader
        if args.task=='binary':
            args.num_classes = 2
            args.ignore_label = 2
            args.cls_wt = [1.0, 1.0]
        elif args.task=='parts':
            args.num_classes = 5
            args.ignore_label = 5
            args.cls_wt = [0.1, 1.0, 1.0, 1.0, 1.0]
        elif args.task=='type':
            args.num_classes = 8
            args.ignore_label = 8
            args.cls_wt = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        train_loader, val_loader = get_dataloader(args.task, batch_size=args.batch_size)
        net_param = {"class_num"   : args.num_classes,
                     "in_chns"     : 3,
                     "bilinear"    : True,
                     "feature_chns": [16, 32, 64, 128, 256],
                     "dropout"     : [0.0, 0.0, 0.3, 0.4, 0.5]}
    elif args.dataset=='covid19_lesion':
        from datasets.covid19_lesion import get_dataloader
        train_loader, val_loader = get_dataloader(args.task, batch_size=args.batch_size)
        args.ignore_label = 2
        args.num_classes = 2
        net_param = {"class_num"   : args.num_classes,
                     "in_chns"     : 1,
                     "bilinear"    : True,
                     "feature_chns": [16, 32, 64, 128, 256],
                     "dropout"     : [0.0, 0.0, 0.3, 0.4, 0.5]}
    else:
        raise NotImplementedError('The dataset is not supported.')

    if args.margin_loss:
        args.margins = loss.calculate_margins(train_loader, args.num_classes)
        

    criterion = loss.get_loss(args)
    net = COPLENet(net_param).cuda()
    optim, scheduler = optimizer.get_optimizer(args, net)

    epoch = 0
    i = 0
    if args.snapshot:
        epoch, mean_iou = optimizer.load_weights(net, optim, scheduler,
                                args.snapshot, args.restore_optimizer)
        if args.restore_optimizer:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()

    while i < args.max_iter and epoch < args.max_epoch:
        i = train(train_loader, net, criterion, optim, epoch, scheduler, args.max_iter)
        #iou_train = iou_on_trainset(train_loader, net, criterion)
        #logging.info('Mean IoU on training set: %f' % iou_train)
        val_loss, per_cls_iou = validate(val_loader, net, criterion, optim, scheduler, epoch+1, i)
        epoch += 1

'''
def iou_on_trainset(train_loader, net, criterion):
    net.eval()
    hist = 0
    error_acc = 0

    for val_idx, data in enumerate(train_loader):
        inputs, gts = data = data
        assert len(inputs.size()) == 4 and len(gts.size()) == 3
        assert inputs.size()[2:] == gts.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gts = inputs.cuda(), gts.cuda()

        with torch.no_grad():
            output = net(inputs)
        del inputs
        assert output.size()[2:] == gts.size()[1:]
        assert output.size()[1] == args.num_classes

        predictions = output.data.max(1)[1].cpu()
        
        hist += fast_hist(predictions.numpy().flatten(), gts.cpu().numpy().flatten(),
                             args.num_classes)
        del gts, output, val_idx, data

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)
    return  mean_iou
'''

def train(train_loader, net, criterion, optim, curr_epoch, scheduler, max_iter):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break
        start_ts = time.time()
        inputs, gts = data
        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gts = inputs.cuda(), gts.cuda()
        optim.zero_grad()
        outputs = net(inputs)
        total_loss = criterion(outputs, gts)
        log_total_loss = total_loss.clone().detach_()
        train_total_loss.update(log_total_loss.item(), batch_pixel_size)

        total_loss.backward()
        optim.step()
        scheduler.step()

        time_meter.update(time.time() - start_ts)

        del total_loss

        curr_iter += 1
        if i % 50 == 49:

            msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                      curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                      optim.param_groups[-1]['lr'], time_meter.avg / args.batch_size)
            logging.info(msg)
            train_total_loss.reset()
            time_meter.reset()
    return curr_iter

def validate(val_loader, net, criterion, optim, scheduler, curr_epoch, curr_iter):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0

    for val_idx, data in enumerate(val_loader):
        inputs, gts = data = data
        assert len(inputs.size()) == 4 and len(gts.size()) == 3
        assert inputs.size()[2:] == gts.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gts = inputs.cuda(), gts.cuda()

        with torch.no_grad():
            output = net(inputs)
        del inputs
        assert output.size()[2:] == gts.size()[1:]
        assert output.size()[1] == args.num_classes
        val_loss.update(criterion(output, gts).item(), batch_pixel_size)

        predictions = output.data.max(1)[1].cpu()
        # Logging
        if val_idx % 20 == 0:
            logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        iou_acc += fast_hist(predictions.numpy().flatten(), gts.cpu().numpy().flatten(),
                             args.num_classes)
        del gts, output, val_idx, data

    per_cls_iou = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, curr_epoch, args.dataset, curr_iter)
    return val_loss.avg, per_cls_iou


if __name__=='__main__':
    main()
