import argparse
import os
import torch
from network.coplenet import COPLENet
from utils.misc import fast_hist
import datasets
import optimizer
import numpy as np
import colorsys
import skimage

parser = argparse.ArgumentParser(description='CopleNet for medical image segmentation')
parser.add_argument('--dataset', type=str, default='robotic_instrument',
                    help='robotic_instrument, covid19_leision')
parser.add_argument('--task', type=str, required=True, help='classification task. `binary` or `parts` for robotic_instrument, and `part1` or `part2` for covid19_leision')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training and validation')
parser.add_argument('--snapshot', type=str, required=True)
parser.add_argument('--dump_imgs', action='store_true', default=False)
parser.add_argument('--method', type=str, default='')


args = parser.parse_args()

torch.backends.cudnn.benchmark = True


def add_color(img, num_classes=32):
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in range(1, 151):
        v = (i - 1) * (137.5 / 360)
        img_color[img == i] = colorsys.hsv_to_rgb(v, 1, 1)
    img_color[img == num_classes] = (1.0, 1.0, 1.0)
    return img_color


def main():
    if args.dataset=='robotic_instrument':
        from datasets.robotic_instrument import get_testloader, RoboticInstrument
        if args.task=='binary':
            num_classes = 2
        elif args.task=='parts':
            num_classes = 5
        elif args.task=='type':
            num_classes = 8
        dataset = RoboticInstrument(args.task, 'test')
        test_loader = get_testloader(args.task, batch_size=args.batch_size)
        net_param = {"class_num"   : num_classes,
                     "in_chns"     : 3,
                     "bilinear"    : True,
                     "feature_chns": [16, 32, 64, 128, 256],
                     "dropout"     : [0.0, 0.0, 0.3, 0.4, 0.5]}
    elif args.dataset=='covid19_lesion':
        from datasets.covid19_lesion import get_testloader, Covid19Dataset
        dataset = Covid19Dataset(args.task, 'test')
        test_loader = get_testloader(args.task, batch_size=args.batch_size)
        num_classes = 2
        net_param = {"class_num"   : num_classes,
                     "in_chns"     : 1,
                     "bilinear"    : True,
                     "feature_chns": [16, 32, 64, 128, 256],
                     "dropout"     : [0.0, 0.0, 0.3, 0.4, 0.5]}
    else:
        raise NotImplementedError('The dataset is not supported.')
    
    net = COPLENet(net_param).cuda()
    optimizer.load_weights(net, None, None, args.snapshot, False)    
    torch.cuda.empty_cache()

    net.eval()
    hist = 0
    predictions = []
    groundtruths = []
    for test_idx, data in enumerate(test_loader):
        inputs, gts = data 
        assert len(inputs.size()) == 4 and len(gts.size()) == 3
        assert inputs.size()[2:] == gts.size()[1:]
        inputs, gts = inputs.cuda(), gts.cuda()
        with torch.no_grad():
            output = net(inputs)
        del inputs
        assert output.size()[2:] == gts.size()[1:]
        assert output.size()[1] == num_classes
        
        prediction = output.data.max(1)[1].cpu()
        predictions.append(output.data.cpu().numpy())
        groundtruths.append(gts.cpu().numpy())
        hist += fast_hist(prediction.numpy().flatten(), gts.cpu().numpy().flatten(),
                             num_classes)
        del gts, output, test_idx, data
    
    predictions = np.concatenate(predictions, axis=0)
    groundtruths = np.concatenate(groundtruths, axis=0)
    if args.dump_imgs:
        assert len(dataset)==predictions.shape[0]
    
        dump_dir = './dump_' + args.dataset + '_' + args.task + '_' + args.method
        os.makedirs(dump_dir, exist_ok=True)
        for i in range(len(dataset)):
            img = skimage.io.imread(dataset.img_paths[i])
            if len(img.shape)==2:
                img = np.stack((img, img, img), axis=2)
            img = skimage.transform.resize(img, (224,336))
            cm = np.argmax(predictions[i,:,:,:], axis=0)
            color_cm = add_color(cm)
            color_cm = skimage.transform.resize(color_cm, (224,336))
            gt = np.asarray(groundtruths[i,:,:], np.uint8)
            color_gt = add_color(gt)
            color_gt = skimage.transform.resize(color_gt, (224,336))
            blend_pred = 0.5 * img + 0.5 * color_cm
            blend_gt = 0.5 * img + 0.5 * color_gt
            blend_pred = np.asarray(blend_pred*255, np.uint8)
            blend_gt = np.asarray(blend_gt*255, np.uint8)
            #skimage.io.imsave(os.path.join(dump_dir, 'img_{:03d}.png'.format(i)), img)
            skimage.io.imsave(os.path.join(dump_dir, 'pred_{:03d}.png'.format(i)), blend_pred)
            skimage.io.imsave(os.path.join(dump_dir, 'gt_{:03d}.png'.format(i)), blend_gt)
            if i > 20:
                break
    
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))    
    id2cat = {i: i for i in range(len(iou))}
    iou_false_positive = hist.sum(axis=1) - np.diag(hist)
    iou_false_negative = hist.sum(axis=0) - np.diag(hist)
    iou_true_positive = np.diag(hist)

    print('IoU:')
    print('label_id      label    IoU    Precision Recall TP       FP      FN      Pixel Acc.')
    for idx, i in enumerate(iou):
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iou_string = '{:5.1f}'.format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.1f}'.format(100 * iou_true_positive[idx] / total_pixels)
        fp = '{:5.1f}'.format(100 * iou_false_positive[idx] / total_pixels)
        fn = '{:5.1f}'.format(100 * iou_false_negative[idx] / total_pixels)
        precision = '{:5.1f}'.format(
            iou_true_positive[idx] / (iou_true_positive[idx] + iou_false_positive[idx]))
        recall = '{:5.1f}'.format(
            iou_true_positive[idx] / (iou_true_positive[idx] + iou_false_negative[idx]))
        pixel_acc = '{:5.1f}'.format(100*acc_cls[idx])
        print('{}    {}   {}  {}     {}  {}   {}   {}   {}'.format(
            idx_string, class_name, iou_string, precision, recall, tp, fp, fn, pixel_acc))

if __name__=='__main__':
    main()
