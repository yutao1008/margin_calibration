"""
Miscellanous Functions
"""

import sys
import re
import os
import shutil
import torch
from datetime import datetime
import logging
from subprocess import call
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tqdm import tqdm

def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_log(prefix, output_dir, date_str):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + date_str +'.log')
    print("Logging :", filename)
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)




def prep_experiment(args, parser):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    ckpt_path = args.ckpt
    exp_name = '{}-{}'.format(args.dataset[:4], 'coplenet')
    args.exp_path = os.path.join(ckpt_path, args.date, args.exp, str(datetime.now().strftime('%m_%d_%H')))

    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iou': 0, 'fwavacc': 0}
    args.last_record = {}
    
    os.makedirs(args.exp_path, exist_ok=True)

    save_log('log', args.exp_path, args.date_str)
    open(os.path.join(args.exp_path, args.date_str + '.txt'), 'w').write(
         str(args) + '\n\n')


def evaluate_eval_for_inference(hist, dataset=None):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    print_evaluate_results(hist, iou, dataset=dataset)
    freq = hist.sum(axis=1) / hist.sum()
    mean_iou = np.nanmean(iou)
    logging.info('mean {}'.format(mean_iou))
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
    return acc, acc_cls, mean_iou, fwavacc



def evaluate_eval(args, net, optimizer, scheduler, val_loss, hist, epoch=0, dataset=None, curr_iter=0):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    print_evaluate_results(hist, iou, dataset)
    freq = hist.sum(axis=1) / hist.sum()
    mean_iou = np.nanmean(iou)
    logging.info('mean {}'.format(mean_iou))
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

    # update latest snapshot
    if 'mean_iou' in args.last_record:
        last_snapshot = 'last_epoch_{}_mean-iou_{:.5f}.pth'.format(
            args.last_record['epoch'], args.last_record['mean_iou'])
        last_snapshot = os.path.join(args.exp_path, last_snapshot)
        try:
            os.remove(last_snapshot)
        except OSError:
            pass
    last_snapshot = 'last_epoch_{}_mean-iou_{:.5f}.pth'.format(epoch, mean_iou)
    last_snapshot = os.path.join(args.exp_path, last_snapshot)
    args.last_record['mean_iou'] = mean_iou
    args.last_record['epoch'] = epoch
    
    torch.save({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'mean_iou': mean_iou,
            'command': ' '.join(sys.argv[1:])
        }, last_snapshot)

    # update best snapshot
    if mean_iou > args.best_record['mean_iou'] :
        # remove old best snapshot
        if args.best_record['epoch'] != -1:
            best_snapshot = 'best_epoch_{}_mean-iou_{:.5f}.pth'.format(
                args.best_record['epoch'], args.best_record['mean_iou'])
            best_snapshot = os.path.join(args.exp_path, best_snapshot)
            assert os.path.exists(best_snapshot), \
                'cant find old snapshot {}'.format(best_snapshot)
            os.remove(best_snapshot)

        
        # save new best
        args.best_record['val_loss'] = val_loss.avg
        args.best_record['epoch'] = epoch
        args.best_record['acc'] = acc
        args.best_record['acc_cls'] = acc_cls
        args.best_record['mean_iou'] = mean_iou
        args.best_record['fwavacc'] = fwavacc

        best_snapshot = 'best_epoch_{}_mean-iou_{:.5f}.pth'.format(
            args.best_record['epoch'], args.best_record['mean_iou'])
        best_snapshot = os.path.join(args.exp_path, best_snapshot)
        shutil.copyfile(last_snapshot, best_snapshot)

    logging.info('-' * 107)
    fmt_str = '[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], ' +\
              '[mean_iou %.5f], [fwavacc %.5f]'
    logging.info(fmt_str % (epoch, val_loss.avg, acc, acc_cls, mean_iou, fwavacc))
    fmt_str = 'best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], ' +\
              '[mean_iou %.5f], [fwavacc %.5f], [epoch %d], '
    logging.info(fmt_str % (args.best_record['val_loss'], args.best_record['acc'],
                            args.best_record['acc_cls'], args.best_record['mean_iou'],
                            args.best_record['fwavacc'], args.best_record['epoch']))
    logging.info('-' * 107)
    return iou





def print_evaluate_results(hist, iou, dataset=None):
    
    id2cat = {i: i for i in range(len(iou))}
    iou_false_positive = hist.sum(axis=1) - np.diag(hist)
    iou_false_negative = hist.sum(axis=0) - np.diag(hist)
    iou_true_positive = np.diag(hist)

    logging.info('IoU:')
    logging.info('label_id      label    IoU    Precision Recall TP     FP    FN')
    for idx, i in enumerate(iou):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iou_string = '{:5.1f}'.format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.1f}'.format(100 * iou_true_positive[idx] / total_pixels)
        fp = '{:5.1f}'.format(
            iou_false_positive[idx] / iou_true_positive[idx])
        fn = '{:5.1f}'.format(iou_false_negative[idx] / iou_true_positive[idx])
        precision = '{:5.1f}'.format(
            iou_true_positive[idx] / (iou_true_positive[idx] + iou_false_positive[idx]))
        recall = '{:5.1f}'.format(
            iou_true_positive[idx] / (iou_true_positive[idx] + iou_false_negative[idx]))
        logging.info('{}    {}   {}  {}     {}  {}   {}   {}'.format(
            idx_string, class_name, iou_string, precision, recall, tp, fp, fn))




class AverageMeter(object):

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


