import numpy as np 
import os

from PIL import Image

import wandb
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

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
    #images_array = make_grid([images[0], images[1], images[2]], 3)
    images_array = make_grid(images, n)

    images = wandb.Image(images_array, caption = caption)
    #wandb.log({"examples": images, 'epoch':epoch})
    
    return images

class RandomCropSame:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img1, img2):
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

class MyDataset_RandCrop(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low, images_high, cropsize = None, tensor_transform = None, flips=None, test=False):
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
        self.rcrop      = RandomCropSame(self.cropsize)

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
        
        # rgb_low  = Image.open(img_low).convert('RGB')
        # rgb_high = Image.open(img_high).convert('RGB')
        
        if self.to_tensor: #transform the image to have the adequate properties
            rgb_low  = self.to_tensor(rgb_low)
            rgb_high = self.to_tensor(rgb_high)

        # stack high and low to do the exact same flip on the two images
        high_and_low = torch.stack((rgb_high, rgb_low))
        if self.flips:
            high_and_low      = self.flips(high_and_low)
            rgb_high, rgb_low = high_and_low #separate again the images

        if self.cropsize: # do random crops of the image
            rgb_high, rgb_low = self.rcrop(rgb_high, rgb_low)


        return rgb_high, rgb_low

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
        else:
            self.rcrop = None 

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
        
        # rgb_low  = Image.open(img_low).convert('RGB')
        # rgb_high = Image.open(img_high).convert('RGB')
        
        if self.to_tensor: #transform the image to have the adequate properties
            rgb_low  = self.to_tensor(rgb_low)
            rgb_high = self.to_tensor(rgb_high)

        # stack high and low to do the exact same flip on the two images
        high_and_low = torch.stack((rgb_high, rgb_low))
        if self.flips:
            high_and_low      = self.flips(high_and_low)
            rgb_high, rgb_low = high_and_low #separate again the images

        if self.cropsize: # do random crops of the image
            if self.random_crop:   
                rgb_high, rgb_low = self.random_crop(rgb_high, rgb_low)
            elif self.center_crop:
                rgb_high, rgb_low = self.center_crop(rgb_high), self.center_crop(rgb_low)

        return rgb_high, rgb_low

class MyDataset(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low, images_high, image_size = (256, 256, 3), test=False, cropsize = 256):
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
        rgb_low  = np.array(Image.open(img_low).convert('RGB'))
        rgb_high = np.array(Image.open(img_high).convert('RGB'))

        h, w, c = self.image_size

        # normalize to [0,1] fp32
        rgb_low = (rgb_low / 255).astype(np.float32)
        assert rgb_low.shape[-1] == 3 # check it is a 3ch image
        
        rgb_high = (rgb_high / 255).astype(np.float32)
        assert rgb_high.shape[-1] == 3 # check it is a 3ch image

        #### Crop the image only if test = False
        if not self.test:
            rgb_low = crop_center(rgb_low, h, w)
            rgb_high = crop_center(rgb_high, h, w)
        
        #### Convert to PyTorch and change to Pytorch format, instead of [H,W,3] to [3,H,W]
        rgb_low = torch.from_numpy(rgb_low)
        rgb_low = rgb_low.permute((2, 0, 1))
        
        rgb_high = torch.from_numpy(rgb_high)
        rgb_high = rgb_high.permute((2, 0, 1))

        return rgb_high, rgb_low  
    
class MyDataset_nocrop(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low, images_high, test=False):
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
        rgb_low  = np.array(Image.open(img_low).convert('RGB'))
        rgb_high = np.array(Image.open(img_high).convert('RGB'))

        # normalize to [0,1] fp32
        rgb_low = (rgb_low / 255).astype(np.float32)
        assert rgb_low.shape[-1] == 3 # check it is a 3ch image
        
        rgb_high = (rgb_high / 255).astype(np.float32)
        assert rgb_high.shape[-1] == 3 # check it is a 3ch image

        #### Crop the image
        
        #### Convert to PyTorch and change to Pytorch format, instead of [H,W,3] to [3,H,W]
        rgb_low = torch.from_numpy(rgb_low)
        rgb_low = rgb_low.permute((2, 0, 1))
        
        rgb_high = torch.from_numpy(rgb_high)
        rgb_high = rgb_high.permute((2, 0, 1))

        return rgb_high, rgb_low  