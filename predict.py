import os
import yaml
import numpy as np
from PIL import Image
import cv2 as cv

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from ptflops import get_model_complexity_info

from archs import Network, NAFNet, Network_v2, Network_v3
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

#load the config file
PATH_CONFIG = './options/test/LOLBlur.yml'
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
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    dilations = opt['network']['dilations'])
elif network == 'NAFNet':
    model = NAFNet(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_nums'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'])

elif network == 'Network_v2':
    model = Network_v2(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    enc_blk_nums_map=opt['network']['enc_blk_nums_map'],
                    middle_blk_num_map=opt['network']['middle_blk_num_map'],
                    spatial = opt['network']['spatial'],
                    dilations = opt['network']['dilations'])

elif network == 'Network_v3':
    model = Network_v3(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    dilations=opt['network']['dilations'])

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
PATH_IMAGES = opt['examples_path']
PATH_RESULTS = opt['results_path']

path_images = [os.path.join(PATH_IMAGES, path) for path in os.listdir(PATH_IMAGES)]

model.eval()

with torch.no_grad():
    
    for path in path_images:
        tensor = path_to_tensor(path).to(device)
        
        output = torch.clamp(model(tensor), 0., 1.)
        print(output.shape, output.dtype, torch.max(output), torch.min(output))
        save_tensor(output, os.path.join(PATH_RESULTS, os.path.basename(path)))
        
print('Finished predictions.')










