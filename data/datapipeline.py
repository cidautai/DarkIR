import numpy as np 
import os

from PIL import Image

import wandb
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.nn as nn

def crop_center(img, cropx=224, cropy=256):
    """
    Given an image, it returns a center cropped version of size [cropx,cropy]
    """
    y,x,c  = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def log_images(images, caption):
    '''
    A function to log the resulting images to wandb.
    '''
    n=len(images)
    images_array = make_grid(images, n)
    images = wandb.Image(images_array, caption = caption)
    
    return images

class CropTo4(nn.Module):
    def __init__(self):
        super(CropTo4, self).__init__()
        

    def forward(self, img1, img2):
        
        #pad the images with zeros if their size is lower than the cropsize
        img1 = self.pad(img1)
        img2 = self.pad(img2)
        _,_, h, w = img1.shape
        crops1 = [TF.crop(img1, 0, 0, h//2, w//2), TF.crop(img1, 0, w//2, h//2, w),
                  TF.crop(img1, h//2, 0, h, w//2),TF.crop(img1, h//2, w//2, h, w)]
        crops2 = [TF.crop(img2, 0, 0, h//2, w//2), TF.crop(img2, 0, w//2, h//2, w),
                  TF.crop(img2, h//2, 0, h, w//2),TF.crop(img2, h//2, w//2, h, w)]

            
        return torch.cat(crops1, dim=0), torch.cat(crops2, dim=0)

    def pad(self, img):
        _,_, h, w = img.shape
        mod_pad_h = (h - h//2) % 2
        mod_pad_w = (w - w//2) % 2
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), mode = 'constant', value = 0)
        
        return img

class RandomCropSame:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img1, img2):
        
        #pad the images with zeros if their size is lower than the cropsize
        if img1.shape[1] <= self.size[0] or img1.shape[2] <= self.size[1]:
            img1 = self.pad(img1)
            img2 = self.pad(img2)
        
        i, j, th, tw = self.get_params(img1, self.size)
            
        return TF.crop(img1, i, j, th, tw), TF.crop(img2, i, j, th, tw)  # Use th and tw here

    def get_params(self, img, output_size):
        h, w = img.shape[1], img.shape[2]
        th, tw = output_size
        
        if w <= tw or h <= th:
            return 0, 0, h, w
        
        # Calculate the starting top-left corner (i, j) such that the entire crop is within the image.
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        
        return i, j, th, tw

    def pad(self, img):
        _, h, w = img.shape
        mod_pad_h = self.size[0] - h
        mod_pad_w = self.size[1] - w
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h))
        
        return img

class MyDataset_Crop(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low, images_high, cropsize = None, tensor_transform = None, flips=None, test=False, crop_type = 'Random'):
        """
        -images_high: list of RGB images of normal-light used for training or testing the model
        - images_low: list of RGB images of low-light used for training or testing the model
        - test: indicates if the dataset is for training (False) or testing (True)
        -image_size: contains the dimension of the final image (H, W, C). This is important
                     to do the propper crop of the image.
        """
        self.imgs_low   = sorted(images_low)
        self.imgs_high  = sorted(images_high)
        self.test       = test
        self.cropsize   = cropsize
        self.to_tensor  = tensor_transform
        self.flips      = flips
        
        if self.cropsize:
            if crop_type   == 'Random':
                
                self.random_crop = RandomCropSame(self.cropsize)
                self.center_crop = None
            elif crop_type == 'Center':
                
                self.center_crop = transforms.CenterCrop(cropsize)  
                self.random_crop = None

    def __len__(self):
        return len(self.imgs_low)

    def __getitem__(self, idx):
        """
        Given a (random) index. The dataloader selects the corresponding image path, and loads the image.
        Then it returns the image, after applying any required transformation.
        """
        
        img_low  = self.imgs_low[idx]
        img_high = self.imgs_high[idx]
        
        # Load the image and convert to numpy array
        rgb_low  = Image.open(img_low).convert('RGB')
        rgb_high = Image.open(img_high).convert('RGB')

        if self.to_tensor: #transform the image to have the adequate properties
            rgb_low  = self.to_tensor(rgb_low)
            rgb_high = self.to_tensor(rgb_high)

        # stack high and low to do the exact same flip on the two images
        high_and_low = torch.stack((rgb_high, rgb_low))
        if self.flips:
            high_and_low      = self.flips(high_and_low)
            rgb_high, rgb_low = high_and_low #separate again the images
        
        # print(rgb_high.shape, rgb_low.shape)
        if self.cropsize: # do random crops of the image
            if self.random_crop:   
                rgb_high, rgb_low = self.random_crop(rgb_high, rgb_low)
            elif self.center_crop:
                rgb_high, rgb_low = self.center_crop(rgb_high), self.center_crop(rgb_low)
    
        return rgb_high, rgb_low
