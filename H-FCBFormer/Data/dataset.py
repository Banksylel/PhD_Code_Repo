import random
from skimage.io import imread

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import numpy as np

# reads images from path, transforms them to correct size and as a list of floats, also applies augmentation
class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
        classes=1,
        backg=True,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine
        self.n_classes = classes
        self.include_background = backg

        
    # separate masks into binary masks for each class
    # this works by taking pixel 255 as the first class with 0 as background and 255 - 1 as the second class with 0 as background
    def separate_masks(self, target, n_classes, include_background=True):
        first_class = 255

        masks = []
        # extract background mask first
        if include_background:
            masks.append(np.where(target == 0, 255, 0))
            n_classes -= 1

        # loop the number of classes and create a new tensor with 0s and first class value
        for i in range(n_classes):
            masks.append(np.where(target == first_class, 255, 0))
            first_class -= 1

        return masks
    
    def combine_masks(self, target, n_classes, include_background=True):

        #background
        if include_background:
            masks = torch.where(target[0] == 255, 0, 0)
            # traverse targets and combine into a single tensor as 0, 1, 2
            for i in range(n_classes - 1):
                masks = torch.where(target[i+1] == 255, i+1, masks)

        else:
            masks = torch.where(target[0] == 255, 1, 0)

            # traverse targets and combine into a single tensor as 0, 1, 2
            for i in range(n_classes - 1):
                masks = torch.where(target[i+1] == 255, i+2, masks)

        # change 255 to 1 and 254 to 2
        # for i in range(n_classes-1):
        #     masks = np.where(masks == first_class, clss_num, masks)
        #     first_class -= 1
        #     clss_num += 1
            
        return masks

    def decode_segmap(self, temp):   # temp is HW np slice
        r = temp.copy() 
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.class_colors[l][0]
            g[temp == l] = self.class_colors[l][1]
            b[temp == l] = self.class_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))   # dummy tensor in np order HWC
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        x, y = imread(input_ID), imread(target_ID)

        y = self.separate_masks(y, self.n_classes, self.include_background)

        x = self.transform_input(x)
        # transform all targets
        for i in range(len(y)):
            y[i] = self.transform_target(y[i])

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                # hflip all targets
                for i in range(len(y)):
                    y[i] = TF.hflip(y[i])

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                # vflip all targets
                for i in range(len(y)):
                    y[i] = TF.vflip(y[i])

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            # affine all targets
            for i in range(len(y)):
                y[i] = TF.affine(y[i], angle, (h_trans, v_trans), scale, shear, fill=0.0)

        # changes pixels to 0.0 if below 125.0 and 255.0 if above 125.0
        for i in range(len(y)):
            y[i] = torch.where(y[i] < 125, 0, 255)
        
        y = self.combine_masks(y, self.n_classes, self.include_background)

        return x.float(), y[0].long()

