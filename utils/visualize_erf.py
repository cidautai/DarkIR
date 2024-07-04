# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
# print(sys.path)
sys.path.append("/home/leadergpu/Python_workspace_Dani/Net-Low-Light-Deblurring")
from archs.nafnet_utils.arch_model import NAFNet
from archs.network_v3 import Network
from torch import optim as optim
from data.dataset_LOLBlur import main_dataset_lolblur

def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='Network_v3', type=str, help='model name')
    parser.add_argument('--weights', default='../models/bests/Network_v3_ln_interpolate_extraDW.pt', type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    parser.add_argument('--data_path', default='/home/leadergpu/Datasets/NBDN_dataset_50k/test', type=str, help='dataset path')
    parser.add_argument('--save_path', default='./erfs/Network_v3.npy', type=str, help='path to save the ERF matrix (.npy file)')
    parser.add_argument('--num_images', default=100, type=int, help='num of images to use')
    args = parser.parse_args()
    return args


def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def main(args):
    #   ================================= transform: resize to 1024x1024
    _, data_loader_val = main_dataset_lolblur(train_path='/home/leadergpu/Datasets/LOLBlur/train',
                                        test_path = '/home/leadergpu/Datasets/LOLBlur/test',
                                        batch_size_train=8,
                                        batch_size_test=1,
                                        flips = True,
                                        verbose=True,
                                        cropsize=256,
                                        num_workers=4,
                                        crop_type='Random')

    if args.model == 'Network_v3':
        model = Network(img_channel=3, 
                    width=32, 
                    middle_blk_num=3, 
                    enc_blk_nums=[1, 2, 3],
                    dec_blk_nums=[3, 1, 1], 
                    residual_layers=1,
                    dilations=[1, 4],
                    extra_depth_wise=True)
    elif args.model == 'NAFNet':
        model = NAFNet(img_channel=3, 
                    width=32, 
                    middle_blk_num=12, 
                    enc_blk_nums=[2, 2, 4, 8],
                    dec_blk_nums=[2, 2, 2, 2])

    else:
        raise ValueError('Unsupported model. Please add it here.')

    if args.weights is not None:
        print('load weights')
        # print(args.weights)
        weights = torch.load(args.weights)
        model.load_state_dict(weights['model_state_dict'])
        print('loaded')

    model.cuda()
    
    if torch.cuda.device_count() > 1:
        print("Usando", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) 
        model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for samples, _ in data_loader_val:

        if meter.count == args.num_images:
            np.save(args.save_path, meter.avg)
            break

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)

    return meter.avg

if __name__ == '__main__':
    args = parse_args()
    erf = main(args)
    # print(erf.shape, np.max(erf), np.min(erf))
    # erf = (erf * 255).astype(np.uint8)
    # img = Image.fromarray(erf)
    # img = img.convert('L')
    # print(img)
    # img.save('./erfs/Network_v3.png')