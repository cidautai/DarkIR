import numpy as np
import random

from scipy.io import loadmat
from scipy import ndimage
from scipy.signal import convolve2d

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import signal
import kornia

from imutils import postprocess_raw, demosaic, save_rgb, plot_all


def augment_kernel(kernel, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    Rotate kernels (or images)
    '''
    if mode == 0:
        return kernel
    elif mode == 1:
        return np.flipud(np.rot90(kernel)) # np.flipud is equivalent to np.flip(m, axis =0)
    elif mode == 2:
        return np.flipud(kernel)
    elif mode == 3:
        return np.rot90(kernel, k=3) # here k stands for the number of 90º rotations applied.
    elif mode == 4:
        return np.flipud(np.rot90(kernel, k=2))
    elif mode == 5:
        return np.rot90(kernel)
    elif mode == 6:
        return np.rot90(kernel, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(kernel, k=3))
    
def apply_custom_filter(inp_img, kernel_):
    return kornia.filters.filter2d(inp_img, kernel_, normalized=True) # this function convolves the filter with the input img

def generate_gkernel(ker_sz=None, sigma=None):
    '''
    ker_sz: an int value 
    sigma : a tuple
    '''
    
    gkern1 = signal.gaussian(ker_sz, std=sigma[0]).reshape(ker_sz, 1)
    gkern2 = signal.gaussian(ker_sz, std=sigma[1]).reshape(ker_sz, 1)
    gkern  = np.outer(gkern1, gkern2)
    return gkern
    
def apply_gkernel(inp_img, ker_sz=5, ksigma_vals=[.05 + i for i in range(5)]):
    """
    Apply uniform gaussian kernel of sizes between 5 and 11.
    """
    # sample for variance
    sigma_val1 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma_val2 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma = (sigma_val1, sigma_val2)
    
    kernel = generate_gkernel(ker_sz, sigma)
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.), kernel
    
def apply_psf(inp_img, kernels):
    """
    Apply PSF
    """
    idx = np.random.choice(np.arange(11), p=[0.15,0.20,0.20,0.0075,0.0075,0.175,0.175,0.05,0.0075,0.0075,0.02])
    kernel = kernels[idx].astype(np.float64)
    kernel = augment_kernel(kernel, mode=random.randint(0, 7))
    ker_sz = 25
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.), kernel

def apply_random_kernel_from_list_of_kernels(inp_img, kernels):
    """
    Apply PSF
    """
    idx = np.random.choice(np.arange(len(kernels)))
    kernel = kernels[idx].astype(np.float64)
    kernel = augment_kernel(kernel, mode=random.randint(0, 7))
    ker_sz = 25
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel).squeeze(0)
    return torch.clamp(blurry, 0., 1.), kernel

def add_blur(inp_img, kernels, plot=False, gkern_szs= [3, 5, 7, 9]):
        
    # sample for kernel size
    ker_sz = gkern_szs[np.random.randint(len(gkern_szs))]
    use_gkernel = random.random() > 0.5
    kernel_type = ''
    
    if use_gkernel:
        kernel_type=f'gaussian_{ker_sz}'
        blurry, kernel = apply_gkernel(inp_img.unsqueeze(0), ker_sz=ker_sz)
    else:
        kernel_type=f'psf'
        blurry, kernel = apply_psf(inp_img.unsqueeze(0), kernels)

    # if plot:
    #     kernelid = np.random.randint(999999)
    #     print ('Kernel', kernelid, kernel.shape, kernel_type)
    #     save_rgb((kernel*255).astype(np.float32), f'kernel_{kernelid}.png')
        
    return blurry

def normalize_to_01(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    # Normalize the tensor to the range [0, 1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    return normalized_tensor

def apply_3Dblur(image, kernel):
    '''
    This function will convolve the image along each of its channels with one specific kernel.
    image: torch.tensor of size [B, C, H, W]
    kernel: torch.tensor of size [B, C, h, w]
    '''
    r, g, b = torch.chunk(image, 3, dim=1)
    r_k, g_k, b_k = torch.chunk(kernel, 3, dim=0)

    r_blur = kornia.filters.filter2d(r, r_k, normalized=False)
    g_blur = kornia.filters.filter2d(g, g_k, normalized=False)
    b_blur = kornia.filters.filter2d(b, b_k, normalized=False)

    image_blur = torch.clamp(torch.cat([r_blur, g_blur, b_blur], dim=1),0, 1)

    return image_blur



if __name__ == "__main__":
    
    from PIL import Image
    import os
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    
    PATH = '../RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene001/gt'
    images = sorted(os.path.join(PATH, img) for img in os.listdir(PATH))
    # print(images[0], images[1])
    #print(images)
    #select only two images
    img1 = Image.open(images[0]).convert('RGB')
    
    PATH_KERNEL = '../Kernel_Estimation_Network/kernels/kernel5'
    list_kernels = [os.path.join(PATH_KERNEL, path) for path in os.listdir(PATH_KERNEL)]
    # print(list_kernels) 
    
    kernels = [np.squeeze(np.load(kernel), axis = -1) for kernel in list_kernels]
    kernels = np.stack(kernels, axis=0)
    print('numpy kernel shape:', kernels.shape)
   
    # kernels_tensor = normalize_to_01(torch.from_numpy(kernels))
    kernels_tensor = torch.from_numpy(kernels)
    print('Info of the kernels tensor:', kernels_tensor.shape, kernels_tensor.dtype, torch.max(kernels_tensor), torch.min(kernels_tensor))
    print('Sum of the elements of kernel:', torch.sum(kernels_tensor[0]))
    
    transform_to_tensor = transforms.ToTensor()
    img_tensor = transform_to_tensor(img1)
    img_tensor_batch = img_tensor.unsqueeze(0)
    print('Info of the img tensor: ',img_tensor_batch.shape, img_tensor_batch.dtype, torch.max(img_tensor_batch), torch.min(img_tensor_batch))
    
    image_blur = apply_3Dblur(img_tensor_batch, kernels_tensor)
    
    NEW_PATH = './images_blur'
    not os.path.isdir(NEW_PATH) and os.mkdir(NEW_PATH)
    
    to_pil = transforms.ToPILImage()
    image_blur = to_pil(image_blur.squeeze(0))
    
    print(image_blur)
    image_blur.save(os.path.join(NEW_PATH, 'blur5.png'))
    img1.save(os.path.join(NEW_PATH, 'sharp.png'))
    
    
    
    
    
    
    
    
    