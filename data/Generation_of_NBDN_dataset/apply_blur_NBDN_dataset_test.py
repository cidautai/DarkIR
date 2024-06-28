'''
This script is for generating blur test images using the NBDN dataset and its kernels.
It has a similar structure to the one that generates the training set.
'''

import numpy as np
import torch 
import hdf5storage
import matplotlib.pyplot as plt
from PIL import Image
import os

#import the relevant functions to make blur and noise
from utils.degradations import simple_deg_simulation
from utils.blur import apply_psf, apply_gkernel, apply_custom_filter
from utils.imutils import save_rgb, plot_all, plot_all_mosaic, convert_to_tensor
import torchvision.transforms as transforms
from utils.blur import augment_kernel
import random


path_folder  = '../../../NBDN_dataset/testset_public/kernel'

path_kernels  = [os.path.join(path_folder, path) for path in os.listdir(path_folder)]


print(len(path_kernels))

kernels = [np.asarray(Image.open(path)) for path in path_kernels]

# this kernels aren't normalized to one
# let's normalize them to 0-1
kernels = [(kernel/255).astype(np.float32) for kernel in kernels]

#now normalize them to total sum equals 1
kernels = [kernel/np.sum(kernel) for kernel in kernels]
print('Number of kernels: ', len(kernels))
print('Kernels properties:')
print(kernels[0].shape, kernels[0].dtype, np.max(kernels[0]), np.min(kernels[0]), np.sum(kernels[0]))

# now we can start opening our path of images
dir_path_images = '../../../NBDN_dataset/testset_public/100_color_imgs_for_test'

path_images = [os.path.join(dir_path_images, path) for path in os.listdir(dir_path_images)]

#we need to order the images, to apply their corresponding kernel
extensions = [os.path.splitext(path) for path in path_images]
ordered = sorted(extensions, key = lambda x: int(os.path.basename(x[0])))
path_images = [order[0]+order[1] for order in ordered]

print('Number of images: ', len(path_images))

trues = [os.path.isfile(file) for file in path_images]
for true in trues:
    if true != True:
        print('Non valid route!')
        
#create the different folders
PATH_IMAGES = '../../../NBDN_dataset_50k'
not os.path.isdir(PATH_IMAGES) and os.mkdir(PATH_IMAGES)

PATH_IMAGES_TEST = os.path.join(PATH_IMAGES, 'test')
not os.path.isdir(PATH_IMAGES_TEST) and os.mkdir(PATH_IMAGES_TEST)
PATH_IMAGES_SHARP = os.path.join(PATH_IMAGES_TEST, 'sharp')
not os.path.isdir(PATH_IMAGES_SHARP) and os.mkdir(PATH_IMAGES_SHARP)
PATH_IMAGES_BLUR = os.path.join(PATH_IMAGES_TEST, 'blur')
not os.path.isdir(PATH_IMAGES_BLUR) and os.mkdir(PATH_IMAGES_BLUR)

#lets prepare some things for the img to tensor transformation
transform_to_tensor = transforms.ToTensor()
to_plot = lambda img: (img.squeeze(0).permute(1, 2, 0).numpy()*255).astype(np.uint8) # get back to image from tensor

'''
Now for each image we will make a folder in the training. For each image we save a blurred and sharp one.

'''

for idx, (path_img, kernel) in enumerate(zip(path_images, kernels), start = 1):
    
    # path_sharp = os.path.join(PATH_IMAGES_SHARP, f'img_{idx}')
    # not os.path.isdir(path_sharp) and os.mkdir(path_sharp)
    
    # path_blur = os.path.join(PATH_IMAGES_BLUR, f'img_{idx}')
    # not os.path.isdir(path_blur) and os.mkdir(path_blur)
    
    img_gt = Image.open(path_img).convert('RGB')
    img_gt_tensor = transform_to_tensor(img_gt).unsqueeze(0)
    
    #we enlarge the dinamic range by 1.2
    img_gt_tensor_save = img_gt_tensor * 1.2
    img_gt = Image.fromarray(to_plot(torch.clamp(img_gt_tensor_save, 0., 1.)))
    img_gt.save(os.path.join(PATH_IMAGES_SHARP, f'{idx}.png'))
    
    print(idx, img_gt_tensor.shape)
    
    kernel_tensor = transform_to_tensor(kernel)
    # print(kernel_tensor.shape)
    new_img = apply_custom_filter(img_gt_tensor, kernel_tensor)
    
    #add the gaussian noise
    noise = torch.normal(mean = 0, std = 0.01, size = new_img.shape)
    new_img = new_img + noise
    #we enlarge the dinamic range by 1.2 
    new_img = new_img * 1.2
    #and finally clip it to [0, 1]
    new_img = torch.clamp(new_img, 0., 1.)

    new_img = Image.fromarray(to_plot(new_img))
    new_img.save(os.path.join(PATH_IMAGES_BLUR, f'{idx}.png'))
    
