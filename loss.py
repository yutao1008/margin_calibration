"""
Loss.py
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lovasz_losses import lovasz_softmax
from torch.autograd import Variable
from torch.autograd import Function
import copy
from tqdm import tqdm


def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    cls_wt: class weights for imbalanced labels
    return: criterion, criterion_val
    """
    if args.cls_wt_loss:
        cls_wt = torch.Tensor(args.cls_wt)
    else:
        cls_wt = None


    if args.ce2d_loss:
        logging.info("Using cross entropy loss 2d")
        criterion = CrossEntropyLoss2d(weight=cls_wt, ignore_index=args.ignore_label).cuda()
    elif args.dice_loss:
        logging.info("Using generalized dice loss")
        criterion = GDiceLossV2().cuda()
    elif args.focal_loss:
        logging.info("Using focal loss")
        criterion = FocalLoss().cuda()
    elif args.tversky_loss:
        logging.info("Using Tversky loss")
        criterion = TverskyLoss().cuda()
    elif args.lovasz_softmax:
        logging.info("Using lovasz softmax loss")
        criterion = LovaszSoftmax(ignore_index=args.ignore_label, per_image=True).cuda()
    elif args.margin_loss:
        logging.info("Using margin loss")
        criterion = Margin_logloss(cls=args.num_classes, margins=args.margins, ignore_index=args.ignore_label).cuda()
    else:
        logging.info("Using default cross entropy loss")
        criterion = nn.CrossEntropyLoss(weight=cls_wt, reduction='mean',
                                       ignore_index=args.ignore_label).cuda()

    return criterion

class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='mean', ignore_index=ignore_index)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        return self.nll_loss(self.logsoftmax(inputs), targets)


class LovaszSoftmax(nn.Module):
    """
    Multi-class Lovasz-Softmax loss
    """
    def __init__(self, ignore_index=255, per_image=False):
        super(LovaszSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, inputs, targets):
        return lovasz_softmax(self.softmax(inputs), targets, per_image=self.per_image, ignore=self.ignore_index)

def calculate_margins(dataloader, num_classes, tau=10, upsilon=1, ignore_index=255):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    print('Calculating per-class margins')
    for sample in tqdm(dataloader):
        y = sample[1]
        y = y.detach().cpu().numpy()
        mask = (y != ignore_index) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    n_pixels = np.sum(z)
    rhoi0s = []
    rho0is = []
    for pixels in z:
        cls_prob = pixels / n_pixels
        bg_pixels = n_pixels - pixels
        rho0i = tau * bg_pixels**0.5 / pixels
        rhoi0 = rho0i * cls_prob * pixels**0.5 / ( upsilon*bg_pixels - cls_prob * bg_pixels**0.5 )
        rhoi0s.append(rhoi0)
        rho0is.append(rho0i) 
    return np.array([rhoi0s,rho0is])
    
class Margin_logloss(nn.Module):
    """
    Our Margin_logloss
    """
    def __init__(self, cls, margins, ignore_index=255):
        super(Margin_logloss, self).__init__()
        self.ignore_index = ignore_index
        self.register_buffer('margins', torch.tensor(margins))
        self.register_buffer('cls', torch.arange(cls))

    def forward(self, logit, target):
        n, c, h, w = logit.size()
        logit = logit.transpose(1,0).contiguous().view(c, -1).t()
        target = target.view(-1)
        logit = logit[target!=self.ignore_index]
        target = target[target!=self.ignore_index]
        max2_score, inds = logit.topk(k=2,dim=1)
        sub_max_inds = inds[:,0].expand(c,-1).t() == self.cls
        sub_max_score = torch.gather(max2_score, 1, sub_max_inds.long())
        score_all = logit - sub_max_score
        #=========================
        margins_all = -self.margins[1].expand(score_all.shape)
        p_margins = torch.gather(self.margins[0],0,target)
        margins_all.scatter_(1, target.unsqueeze(1), p_margins.unsqueeze(1))
        score_all -= margins_all
        #=========================
        loss_mean = F.binary_cross_entropy_with_logits(score_all, (margins_all>0).float(), pos_weight=torch.tensor(c))
        return loss_mean

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)

class GDiceLossV2(nn.Module):
    #def __init__(self, apply_nonlin=None, smooth=1e-5):
    def __init__(self, smooth=1e-5):
        """
        Generalized Dice V2;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        softmax_output = nn.Softmax(dim=1)(net_output)

        input = flatten(softmax_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = 1. / (target_sum * target_sum).clamp(min=self.smooth)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.smooth)

class TverskyLoss(nn.Module):
    def __init__(self, smooth=1., alpha=0.3, beta=0.7, square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()
        self.square = square
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        axes = list(range(2, len(shp_x)))
        x = nn.Softmax(dim=1)(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        tversky = tversky.mean()
        return 1.0 - tversky

def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class  = list(x.size())[1]
    if(tensor_dim == 5):
        x_perm  = x.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        x_perm  = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))   
    y = torch.reshape(x_perm, (-1, num_class)) 
    return y 

class NRDiceLoss(nn.Module):
    '''
    Noise robust dice loss.
    See: A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions from CT Images, TMI, 
    '''
    def __init__(self, gamma=1.5):
        super(NRDiceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        predict = nn.Softmax(dim = 1)(input)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(target)
        numerator = torch.abs(predict - soft_y)
        numerator = torch.pow(numerator, self.gamma)
        numerator = torch.sum(numerator, dim = 0)
        y_vol = torch.sum(soft_y,  dim = 0)
        p_vol = torch.sum(predict, dim = 0)
        loss = (numerator + 1e-5) / (y_vol + p_vol + 1e-5)
        return torch.mean(loss) 

    
    