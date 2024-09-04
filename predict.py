import os
import yaml
import numpy as np
from PIL import Image
import cv2 as cv

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from ptflops import get_model_complexity_info

from archs import Network, NAFNet
from options.options import parse

#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    
    return img
def normalize_tensor(tensor):
    
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    
    tensor = tensor.squeeze(0)
    # tensor = normalize_tensor(tensor)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

#load the config file
PATH_CONFIG = './options/predict/LOLBlur.yml'
opt = parse(PATH_CONFIG)

# define some parameters based on the run we want to make
device = torch.device('cuda')

#selected network
network = opt['network']['name']

# PATH_MODEL = opt['save']['best']
print(network)
# define the network
if network == 'Network':
    model = Network(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num_enc=opt['network']['middle_blk_num_enc'],
                    middle_blk_num_dec=opt['network']['middle_blk_num_dec'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    dilations=opt['network']['dilations'],
                    extra_depth_wise=opt['network']['extra_depth_wise'],
                    ksize=opt['network']['ksize'],
                    side_out=False)
elif network == 'NAFNet':
    model = NAFNet(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'])
else:
    raise NotImplementedError

print(opt['save']['best'])
checkpoints = torch.load(opt['save']['best'])
# print(checkpoints)
model.load_state_dict(checkpoints['model_state_dict'])
model = model.to(device)

#calculate MACs and number of parameters
macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat = False)
print('Computational complexity: ', macs)
print('Number of parameters: ', params)

#load the images and transform to torch
PATH_IMAGES_LOLBLUR = opt['LOLBlur']['inputs_path']
PATH_RESULTS_LOLBLUR = opt['LOLBlur']['results_path']

PATH_IMAGES_REALBLUR = opt['RealBlur']['inputs_path']
PATH_RESULTS_REALBLUR = opt['RealBlur']['results_path']

not os.path.isdir('./results') and os.mkdir('./results')
not os.path.isdir(PATH_RESULTS_LOLBLUR) and os.mkdir(PATH_RESULTS_LOLBLUR)
not os.path.isdir(PATH_RESULTS_REALBLUR) and os.mkdir(PATH_RESULTS_REALBLUR)

path_images_lolblur = [os.path.join(PATH_IMAGES_LOLBLUR, path) for path in os.listdir(PATH_IMAGES_LOLBLUR)]
path_images_realblur = [os.path.join(PATH_IMAGES_REALBLUR, path) for path in os.listdir(PATH_IMAGES_REALBLUR)]

model.eval()

with torch.no_grad():
    
    for path in path_images_lolblur:
        tensor = path_to_tensor(path).to(device)
        _, _, H, W = tensor.shape
        tensor = pad_tensor(tensor)
        output = torch.clamp(model(tensor), 0., 1.)
        output = output[:,:, :H, :W]
        print(output.shape, output.dtype, torch.max(output), torch.min(output))
        save_tensor(output, os.path.join(PATH_RESULTS_LOLBLUR, os.path.basename(path)))
    
    for path in path_images_realblur:
        tensor = path_to_tensor(path).to(device)
        _, _, H, W = tensor.shape
        tensor = pad_tensor(tensor)
        output = torch.clamp(model(tensor), 0., 1.)
        output = output[:,:, :H, :W]
        print(output.shape, output.dtype, torch.max(output), torch.min(output))
        save_tensor(output, os.path.join(PATH_RESULTS_REALBLUR, os.path.basename(path)))
print('Finished predictions.')










