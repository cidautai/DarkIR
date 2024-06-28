import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import cv2

import torch
import torch.nn.functional as F


def save_rgb (img, filename):
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def flip(raw_img, flip):
    if flip == 3:
        raw_img = np.rot90(raw_img, k=2)
    elif flip == 5:
        raw_img = np.rot90(raw_img, k=1)
    elif flip == 6:
        raw_img = np.rot90(raw_img, k=3)
    else:
        pass
    return raw_img


def pack_raw(im):
    """
    Pack RAW image from (h,w,1) to (h/2 , w/2, 4)
    """

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    ## R G G B
    out = np.concatenate((im[0:H:2,0:W:2,:], 
                          im[0:H:2,1:W:2,:],
                          im[1:H:2,0:W:2,:],
                          im[1:H:2,1:W:2,:]), axis=2)

    return out


def demosaic (raw, pattern='RGGB'):
    """Simple demosaicing to visualize RAW images
    Inputs:
     - raw: (h,w,4) RAW RGGB image normalized [0..1] as float32
    Returns: 
     - Simple Avg. Green Demosaiced RAW image with shape (h*2, w*2, 3)
    """
    
    assert raw.shape[-1] == 4
    shape = raw.shape
    
    c1 = raw[:,:,0]
    c2 = raw[:,:,1]
    c3 = raw[:,:,2]
    c4 = raw[:,:,3]
    
    if pattern == 'RGGB':
        red= c1; green_red=c2; green_blue=c3; blue=c4
        avg_green  = (green_red + green_blue) / 2
        
    elif pattern == 'GBRG':
        red= c3; green_red=c1; green_blue=c4; blue=c2
        avg_green  = (green_red + green_blue) / 2
        
    elif pattern == 'GRBG':
        red= c2; green_red=c1; green_blue=c4; blue=c3
        avg_green  = (green_red + green_blue) / 2
        
    else:
        print ('Wrong pattern', pattern, 'only RGGB / GRBG are supported.')
        return 0
        
    image      = np.stack((red, avg_green, blue), axis=-1)
    image      = cv2.resize(image, (shape[1]*2, shape[0]*2))
    return image


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def mosaic(rgb):
    """Extracts RGGB Bayer planes from an RGB image."""
    
    assert rgb.shape[-1] == 3
    shape = rgb.shape
    
    red        = rgb[0::2, 0::2, 0]
    green_red  = rgb[0::2, 1::2, 1]
    green_blue = rgb[1::2, 0::2, 1]
    blue       = rgb[1::2, 1::2, 2]
    
    image = np.stack((red, green_red, green_blue, blue), axis=-1)
    return image


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return np.maximum(image, 1e-8) ** 2.2


def gamma_compression(image):
    """Converts from linear to gamma space."""
    return np.maximum(image, 1e-8) ** (1.0 / 2.2)


def tonemap(image):
    """Simple S-curved global tonemap"""
    return (3*(image**2)) - (2*(image**3))


def postprocess_raw(raw):
    """Simple post-processing to visualize demosaic RAW imgaes
    Input:  (h,w,3) RAW image normalized
    Output: (h,w,3) post-processed RAW image
    """
    raw = gamma_compression(raw)
    raw = tonemap(raw)
    raw = np.clip(raw, 0, 1)
    return raw


def downsample_raw(raw):
    """
    Downsamples a 4-channel packed RAW image by a factor of 2.
    The input raw should be a [H/2, W/2, 4] tensor -- with respect to its mosaiced version [H,w]
    Output is a [H/4, W/4, 4] tensor, preserving the RGGB pattern.
    """

    # Ensure the image is in [B, C, H, W] format for PyTorch operations
    # raw_image_4channel = raw.permute(2, 0, 1).unsqueeze(0)
    
    # Apply average pooling over a 2x2 window for each channel
    downsampled_image = F.avg_pool2d(raw, kernel_size=2, stride=2, padding=0)
    
    # Rearrange back to [H/4, W/4, 4] format
    downsampled_image = downsampled_image.squeeze(0).permute(1, 2, 0)
    
    return downsampled_image


def convert_to_tensor(image):
    """
    Checks if the input image is a numpy array or a tensor.
    If it's a numpy array, converts it to a tensor.
    
    Parameters:
    - image: The input image, can be either a numpy array or a tensor.
    
    Returns:
    - A PyTorch tensor of the image.
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to tensor
        image_tensor = torch.from_numpy(image.copy())
        # If the image is in HxWxC format, convert it to CxHxW format expected by PyTorch
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        # If it's already a tensor, just return it
        image_tensor = image
    else:
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")

    return image_tensor

def plot_all (images, axis='off', figsize=(16, 8)):
    
    fig = plt.figure(figsize=figsize, dpi=80)
    nplots = len(images)
    for i in range(nplots):
        plt.subplot(1,nplots,i+1)
        plt.axis(axis)
        plt.imshow(images[i])
    plt.show()
    
    
def plot_all_mosaic(images, axis='off', figsize=(16, 8), plots_per_row=3, columns=2, name = 'Image'):
    fig = plt.figure(figsize=figsize, dpi=80)
    nplots = len(images)
    
    for idx, image in enumerate(images):
        ax = fig.add_subplot(columns, plots_per_row, idx + 1)
        ax.axis(axis)
        ax.imshow(image)
        ax.set_title(f'{name} {idx+1}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        
        # Remove spacing between images
        plt.subplots_adjust(wspace=0.01, hspace=0.15)
    
    plt.show()