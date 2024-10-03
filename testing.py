import numpy as np
import os, sys
import time
import wandb
from tqdm import tqdm

# PyTorch library
import torch
import torch.optim

from data.datasets.datapipeline import *
from archs import create_model, create_optim_scheduler, resume_adapter
from losses.loss import SSIM
from data import create_data
from options.options import parse
from lpips import LPIPS

path_options = './options/train/Finetune.yml'


# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need.
print(os.path.isfile(path_options))
opt = parse(path_options)
# print(opt)
# define some parameters based on the run we want to make
device = torch.device('cuda')

#parameters for saving model
PATH_MODEL     = opt['save']['path_model']
PATH_ADAPTER   = opt['save']['path']

# LOAD THE DATALOADERS
train_loader, test_loader = create_data(opt['datasets'])

# DEFINE NETWORK, SCHEDULER AND OPTIMIZER
model, macs, params = create_model(opt['network'], cuda = opt['device']['cuda'])

# save this stats into opt to upload to wandb
opt['macs'] = macs
opt['params'] = params

# save this stats into opt to upload to wandb
opt['macs'] = macs
opt['params'] = params

#-----------
# adapter = torch.load(PATH_ADAPTER)
# adapter = adapter['model_state_dict']
# for name, param in adapter.items():
#     if 'adapter' not in name:
#         print(name) 
# # for name, param in adapter.items():
# #     print(name)

# sys.exit()
#-----------

# define the optimizer
optim, scheduler = create_optim_scheduler(opt['train'], model)

model, optim, scheduler, start_epochs = resume_adapter(model, optim, scheduler, path_adapter=PATH_ADAPTER, 
                                                       path_model = PATH_MODEL, resume=True)

# # if resume load the weights
# checkpoints_model = torch.load(PATH_MODEL)
# checkpoints_adapter = torch.load(PATH_ADAPTER)

# # first load the weights of the baseline model
# model = load_weights(model, old_weights = checkpoints_model['model_state_dict'])
# # Then, load the weights of the adapter
# model = load_weights(model, old_weights=checkpoints_adapter['model_state_dict'])
# # print(checkpoints_model['model_state_dict'].keys())

#---------------------------------------------------------------------------------------------------
# DEFINE METRICS

calc_SSIM = SSIM(data_range=1.)
calc_LPIPS = LPIPS(net = 'vgg').to(device)

valid_psnr = []
valid_ssim = []
valid_lpips = []

model.eval()
# Now we need to go over the test_loader and evaluate the results of the epoch
for high_batch_valid, low_batch_valid in tqdm(test_loader):

    high_batch_valid = high_batch_valid.to(device)
    low_batch_valid = low_batch_valid.to(device)

    with torch.no_grad():
        enhanced_batch_valid = model(low_batch_valid, use_adapter=False)
        # loss
        valid_loss_batch = torch.mean((high_batch_valid - enhanced_batch_valid)**2)
        # PSNR (dB) metric
        valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))
        valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
        valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
        
        
    valid_psnr.append(valid_psnr_batch.item())
    valid_ssim.append(valid_ssim_batch.item())
    valid_lpips.append(torch.mean(valid_lpips_batch).item())

print(f'PSNR validation value: {np.mean(valid_psnr)}')
print(f'SSIM validation value: {np.mean(valid_ssim)}')
print(f'LPIPS validation value: {np.mean(valid_lpips)}')