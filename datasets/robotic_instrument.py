"""
Robotic Instrument Dataset Loader
"""
import imageio
from torch.utils import data
import os
import numpy as np
from PIL import Image
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

root = '/media/DataDisk/data/robotic_instrument/training'

# All images using left_frames

mask_dir_mappings = {
    'instrument_dataset_1' : ['Left_Prograsp_Forceps_labels', 'Maryland_Bipolar_Forceps_labels', 'Other_labels', 'Right_Prograsp_Forceps_labels'], 
    'instrument_dataset_2' : ['Left_Prograsp_Forceps_labels', 'Other_labels', 'Right_Prograsp_Forceps_labels'],
    'instrument_dataset_3' : ['Left_Large_Needle_Driver_labels', 'Right_Large_Needle_Driver_labels'],
    'instrument_dataset_4' : ['Large_Needle_Driver_Left_labels', 'Large_Needle_Driver_Right_labels', 'Prograsp_Forceps_labels'],
    'instrument_dataset_5' : ['Bipolar_Forceps_labels', 'Grasping_Retractor_labels', 'Vessel_Sealer_labels'],
    'instrument_dataset_6' : ['Left_Large_Needle_Driver_labels', 'Monopolar_Curved_Scissors_labels', 'Prograsp_Forceps', 'Right_Large_Needle_Driver_labels'],
    'instrument_dataset_7' : ['Left_Bipolar_Forceps', 'Right_Vessel_Sealer'],
    'instrument_dataset_8' : ['Bipolar_Forceps_labels', 'Left_Grasping_Retractor_labels', 'Monopolar_Curved_Scissors_labels', 'Right_Grasping_Retractor_labels']
}

parts_mapping = {
    "Background": 0,
    "Shaft": 10,
    "Wrist": 20,
    "Claspers": 30,
    "Probe": 40
}

type_mapping = {
    "Bipolar_Forceps": 1,
    "Prograsp_Forceps": 2,
    "Large_Needle_Driver": 3,
    "Vessel_Sealer": 4,
    "Grasping_Retractor": 5,
    "Monopolar_Curved_Scissors": 6,
    "Other": 7
}

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

'''
def make_type_labels():
    for i in range(1, 9):
        subset = 'instrument_dataset_%d' % i 
        print('Processing subset %s' % subset)
        os.makedirs(os.path.join(root, subset, 'ground_truth', 'type_labels'), exist_ok=True)

        for j in range(225):
            img_path = os.path.join(root, subset, 'left_frames/frame{:03d}.png'.format(j))
            img = imageio.imread(img_path)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for mask_dir in mask_dir_mappings[subset]:
                mask_path = os.path.join(root, subset, 'ground_truth', mask_dir, 'frame{:03d}.png'.format(j))  
                for type_name in type_mapping.keys():
                    label = type_mapping[type_name]
                    if type_name in mask_dir:
                        tmp_mask = imageio.imread(mask_path)
                        if len(tmp_mask.shape)==3:
                            tmp_mask = tmp_mask[:,:,0]
                        mask[np.where(tmp_mask>0)] = label            
            target_path = os.path.join(root, subset, 'ground_truth', 'type_labels', 'frame{:03d}.png'.format(j))
            imageio.imwrite(target_path, mask)

def make_binary_labels():
    for i in range(1, 9):
        subset = 'instrument_dataset_%d' % i 
        print('Processing subset %s' % subset)
        os.makedirs(os.path.join(root, subset, 'ground_truth', 'binary_labels'), exist_ok=True)
        for j in range(225):
            img_path = os.path.join(root, subset, 'left_frames/frame{:03d}.png'.format(j))
            img = imageio.imread(img_path)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for mask_dir in mask_dir_mappings[subset]:
                mask_path = os.path.join(root, subset, 'ground_truth', mask_dir, 'frame{:03d}.png'.format(j))  
                tmp_mask = imageio.imread(mask_path)
                if len(tmp_mask.shape)==3:
                    mask += tmp_mask[:,:,0]
                else:
                    mask += tmp_mask
            mask[np.where(mask>0)] = 255
            mask_path = os.path.join(root, subset, 'ground_truth', 'binary_labels', 'frame{:03d}.png'.format(j))
            imageio.imwrite(mask_path, mask)

def make_parts_labels():
    for i in range(1, 9):
        subset = 'instrument_dataset_%d' % i 
        print('Processing subset %s' % subset)
        os.makedirs(os.path.join(root, subset, 'ground_truth', 'parts_labels'), exist_ok=True)
        for j in range(225):
            img_path = os.path.join(root, subset, 'left_frames/frame{:03d}.png'.format(j))
            img = imageio.imread(img_path)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for mask_dir in mask_dir_mappings[subset]:
                mask_path = os.path.join(root, subset, 'ground_truth', mask_dir, 'frame{:03d}.png'.format(j))  
                tmp_mask = imageio.imread(mask_path)
                if len(tmp_mask.shape)==3:
                    mask += tmp_mask[:,:,0]
                else:
                    mask += tmp_mask
            mask[np.where(mask>40)] = 0
            mask_path = os.path.join(root, subset, 'ground_truth', 'parts_labels', 'frame{:03d}.png'.format(j))
            imageio.imwrite(mask_path, mask)
'''

class RoboticInstrument(data.Dataset):
    def __init__(self, task, mode, dump_imgs=False):
        assert task in {'binary', 'type', 'parts'}
        assert mode in {'train', 'val', 'test'}
        self.task = task
        self.mode = mode
        self.dump_imgs = dump_imgs
        self.size = 768
        
        if mode=='train':
            idx_range = range(150)
        elif mode=='val':
            idx_range = range(150, 175)
        else:
            idx_range = range(175, 225)
        #idx_range = range(175) if mode=='train' else range(175,225)
        self.img_paths = []
        self.mask_paths = []
        for i in range(1, 9):
            img_list = [os.path.join(root, 'instrument_dataset_%d' % i, 'left_frames/frame{:03d}.png'.format(j)) for j in idx_range]
            self.img_paths.extend(img_list) 
            mask_list = [os.path.join(root, 'instrument_dataset_%d' % i, 'ground_truth/%s_labels' % task, 'frame{:03d}.png'.format(j)) for j in idx_range]
            self.mask_paths.extend(mask_list)
        
        joint_transform_list = [joint_transforms.Resize(self.size),
                                joint_transforms.RandomHorizontallyFlip()]
        self.train_joint_transform = joint_transforms.Compose(joint_transform_list)
        self.val_joint_transform = joint_transforms.Resize(self.size)
        train_input_transform = []
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.25)]
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize(*mean_std)]
        self.train_input_transform = standard_transforms.Compose(train_input_transform)

        self.val_input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        self.target_transform = extended_transforms.MaskToTensor()
    
    def __getitem__(self, index):
        img_path, mask_path = self.img_paths[index], self.mask_paths[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        if self.dump_imgs:
            outdir = 'dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            img_name = os.path.basename(img_path)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            img.save(out_img_fn)
            mask.save(out_msk_fn)
        img, mask = self.train_joint_transform(img, mask) if self.mode=='train' else self.val_joint_transform(img, mask) 
        img = self.train_input_transform(img) if self.mode=='train' else self.val_input_transform(img)
        mask = self.target_transform(mask)
        if self.task=='binary':
            mask /= 255
        elif self.task=='parts':
            mask /= 10
        return img, mask

    def __len__(self):
        return len(self.img_paths)

def get_dataloader(task, batch_size, num_workers=2):
    train_set = RoboticInstrument(task, mode='train')
    val_set = RoboticInstrument(task, mode='val')
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False, drop_last=False)
    return train_loader, val_loader

def get_testloader(task, batch_size, num_workers=2):
    test_set = RoboticInstrument(task, mode='test')
    test_loader = DataLoader(test_set, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False, drop_last=False)
    return test_loader
