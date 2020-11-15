# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
#import math
#import numbers
import numpy as np
from glob import glob
#import json
#import SimpleITK as sitk
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

root = '/media/DataDisk/data/covid19_lesion'

'''
def load_nifty_volume_as_4d_array(filename):
    """Read a nifty image and return a dictionay storing data array, spacing and direction
    output['data_array'] 4d array with shape [C, D, H, W]
    output['spacing']    a list of spacing in z, y, x axis 
    output['direction']  a 3x3 matrix for direction
    """
    img_obj    = sitk.ReadImage(filename)
    data_array = sitk.GetArrayFromImage(img_obj)
    origin     = img_obj.GetOrigin()
    spacing    = img_obj.GetSpacing()
    direction  = img_obj.GetDirection()
    shape = data_array.shape
    if(len(shape) == 4):
        assert(shape[3] == 1) 
    elif(len(shape) == 3):
        data_array = np.expand_dims(data_array, axis = 0)
    else:
        raise ValueError("unsupported image dim: {0:} 🙊".format(len(shape)))
    output = {}
    output['data_array'] = data_array
    output['origin']     = origin
    output['spacing']    = (spacing[2], spacing[1], spacing[0])
    output['direction']  = direction
    return output

class Resize(object):
    """
    Resize image to exact size of crop
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask, pos=None):
        assert img.size == mask.size
        w, h = img.size
        if pos is not None:
            if (w == h and w == self.size):
                return img, mask, pos
            pos = pos[0].resize(self.size, Image.NEAREST), pos[1].resize(self.size, Image.NEAREST)
            return (img.resize(self.size, Image.BICUBIC),
                    mask.resize(self.size, Image.NEAREST), pos)

        if (w == h and (w, h) == self.size):
            return img, mask
        return (img.resize(self.size, Image.BICUBIC),
                mask.resize(self.size, Image.NEAREST))

def process_export_nifty_data(task):
    assert task in {'part1', 'part2'}
    if task=='part1':
        case_range = range(1,71)
    elif task=='part2':
        case_range = range(1,51)
    img_paths = [os.path.join(root, 'part1/image/case_%d.nii.gz' % i) for i in case_range]
    mask_paths = [os.path.join(root, 'part1/label/case_%d.nii.gz' % i) for i in case_range]
    for case, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):    
        img_dict, mask_dict = load_nifty_volume_as_4d_array(img_path), load_nifty_volume_as_4d_array(mask_path)    
        os.makedirs('./covid19_lesion/%s/image/case_%d' % (task, case+1), exist_ok=True)    
        os.makedirs('./covid19_lesion/%s/label/case_%d' % (task, case+1), exist_ok=True)    
        imgs = img_dict['data_array']
        masks = mask_dict['data_array'] 
        for i in range(imgs.shape[1]):
            img = Image.fromarray(imgs[0,i,:,:]) 
            mask = Image.fromarray(masks[0,i,:,:])
            img, mask = Resize((336,224))(img,mask)
            img = np.asarray(np.asarray(img)*255, np.uint8)
            mask = np.asarray(np.asarray(mask)*255, np.uint8)
            if np.max(mask) > 0:
                img = Image.fromarray(img)
                mask = Image.fromarray(mask)
                img.save('./covid19_lesion/%s/image/case_%d/%d.png' % (task, case+1, i)) 
                mask.save('./covid19_lesion/%s/label/case_%d/%d.png' % (task, case+1, i)) 
'''

class Covid19Dataset(Dataset):
    def __init__(self, task, mode):
        assert task in {'part1', 'part2'}
        assert mode in {'train', 'val', 'test'}
        self.task = task
        self.mode = mode
        if task=='part1':
            #train_ids = range(1, 57)
            #val_ids = range(57,71)
            train_ids = range(1, 41)
            val_ids = range(41, 56)
            test_ids = range(56, 71)
        elif task=='part2':
            #train_ids = range(1, 41)
            #val_ids = range(41, 51)
            train_ids = range(1, 31)
            val_ids = range(31, 41)
            test_ids = range(41, 51)
        if mode=='train':
            img_dirs = [os.path.join(root, task, 'image/case_%d' % i) for i in train_ids]
            mask_dirs = [os.path.join(root, task, 'label/case_%d' % i) for i in train_ids]
        elif mode=='val':
            img_dirs = [os.path.join(root, task, 'image/case_%d' % i) for i in val_ids]
            mask_dirs = [os.path.join(root, task, 'label/case_%d' % i) for i in val_ids]
        else:
            img_dirs = [os.path.join(root, task, 'image/case_%d' % i) for i in test_ids]
            mask_dirs = [os.path.join(root, task, 'label/case_%d' % i) for i in test_ids]
        self.img_paths = []
        self.mask_paths = []
        for img_dir, mask_dir in zip(img_dirs, mask_dirs):
            self.img_paths.extend(sorted(glob(img_dir + '/*.png')))
            self.mask_paths.extend(sorted(glob(mask_dir + '/*.png')))
        

    def __getitem__(self, index):
        img_path, mask_path = self.img_paths[index], self.mask_paths[index]
        img = torch.Tensor(np.asarray(Image.open(img_path), 'float32')/255-0.5)  # image-0.5 is the centering
        img = torch.unsqueeze(img, 0)
        mask = torch.LongTensor(np.asarray(Image.open(mask_path), 'long')/255) 
        return img, mask

    def __len__(self):
        return len(self.img_paths)




def get_dataloader(task, batch_size, num_workers=2):
    train_set = Covid19Dataset(task, mode='train')
    val_set = Covid19Dataset(task, mode='val')
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False, drop_last=False)
    return train_loader, val_loader

def get_testloader(task, batch_size, num_workers=2):
    test_set = Covid19Dataset(task, mode='test')
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=num_workers, shuffle=False, drop_last=False)
    return test_loader



    