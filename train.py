import numpy as np
import os
from glob import glob
from collections import defaultdict
import time
import wandb
from pathlib import Path
from tqdm import tqdm

# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim

from data.datapipeline import *
from archs import Network
from losses.loss import MSELoss, PerceptualLoss, L1Loss, CharbonnierLoss, SSIMloss
from data.dataset_NBDN import main_dataset_nbdn
from options.options import parse
from torch.optim.lr_scheduler import CosineAnnealingLR
from ptflops import get_model_complexity_info

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need.
path_options = '/home/danfei/Python_Workspace/deblur/NAFNet_Fourllie/options/train/NBDN.yml'
print(os.path.isfile(path_options))
opt = parse(path_options)

# define some parameters based on the run we want to make
device = torch.device('cuda')

#parameters of the network
network = opt['network']['name']

#parameters for saving model
PATH_MODEL     = opt['save']['path']
NEW_PATH_MODEL = opt['save']['path']

#---------------------------------------------------------------

wandb_log = opt['wandb']['init']  # flag if we want to output the results in wandb
resume_training = opt['resume_training']  # flag if we want to resume training

start_epochs = 0
last_epochs = opt['train']['epochs']


#load the dataloaders

train_loader, test_loader = main_dataset_nbdn(train_path=opt['datasets']['train']['train_path'],
                                              test_path = opt['datasets']['val']['test_path'],
                                              batch_size_train=opt['datasets']['train']['batch_size_train'],
                                              batch_size_test=opt['datasets']['val']['batch_size_test'],
                                              flips = opt['datasets']['train']['flips'],
                                              verbose=opt['datasets']['train']['verbose'],
                                              cropsize=opt['datasets']['train']['cropsize'],
                                              num_workers=opt['datasets']['train']['n_workers'],
                                              crop_type=opt['datasets']['train']['crop_type'])



# branch = block_branch_try(3, 2, FFN_Expand = 2, dilation = 1, drop_out_rate = 0)
# print('Defining model')
if network == 'Network':
    model = Network(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_nums'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'])
else:
    raise NotImplementedError
model = model.to(device)

#calculate MACs and number of parameters
macs, params = get_model_complexity_info(model, (3, 256, 256))
print('Computational complexity: ', macs)
print('Number of parameters: ', params)

# save this stats into opt to upload to wandb
opt['macs'] = macs
opt['params'] = params

# define the optimizer
optim = torch.optim.AdamW(model.parameters(), lr = opt['train']['lr_initial'],
                          weight_decay = opt['train']['weight_decay'],
                          betas = opt['train']['betas'])

# Initialize the cosine annealing scheduler
# we want the cycle of iterations to
# T_max = len(train_loader) // opt['datasets']['train']['batch_size_train'] * last_epochs
# be the same as the total number of iterations
if opt['train']['lr_scheme'] == 'CosineAnnealing':
    scheduler = CosineAnnealingLR(optim, T_max=last_epochs, eta_min=opt['train']['eta_min'])
else: 
    raise NotImplementedError

# if resume load the weights
if resume_training:
    checkpoints = torch.load(PATH_MODEL)
    model.load_state_dict(checkpoints['model_state_dict'])
    optim.load_state_dict(checkpoints['optimizer_state_dict']),
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    start_epochs = checkpoints['epoch']

# log into wandb
if wandb_log:
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project=opt['wandb']['project'], entity=opt['wandb']['entity'], 
        name=opt['wandb']['name'], save_code=opt['wandb']['save_code'],
        config = opt
    )




# we define the losses
if opt['train']['pixel_criterion'] == 'l1':
    pixel_loss = L1Loss()
elif opt['train']['pixel_criterion'] == 'l2':
    pixel_loss = MSELoss()
elif opt['train']['pixel_criterion'] == 'Charbonnier':
    pixel_loss = CharbonnierLoss()
else:
    raise NotImplementedError

SSIM = SSIMloss()

for epoch in tqdm(range(start_epochs, last_epochs)):

    start_time = time.time()
    train_loss = []
    train_psnr = []       # loss and PSNR of the enhanced-light image
    train_og_loss = []
    train_og_psnr = []  # loss and PSNR of the low-light image
    train_ssim    = []

    if test_loader:
        valid_loss = []
        valid_psnr = []
        valid_ssim = []
    model.train()
    optim_loss = 0

    for high_batch, low_batch in train_loader:

        # Move the data to the GPU
        high_batch = high_batch.to(device)
        low_batch = low_batch.to(device)

        optim.zero_grad()
        # Feed the data into the model
        enhanced_batch = model(low_batch)

        # calculate loss function to optimize
        l_pixel = pixel_loss(enhanced_batch, high_batch)
        optim_loss = l_pixel

        # Calculate loss function for the PSNR
        loss = torch.mean((high_batch - enhanced_batch)**2)
        og_loss = torch.mean((high_batch - low_batch)**2)

        # and PSNR (dB) metric
        psnr = 20 * torch.log10(1. / torch.sqrt(loss))
        og_psnr = 20 * torch.log10(1. / torch.sqrt(og_loss))

        #calculate ssim metric
        ssim = SSIM(enhanced_batch, high_batch)

        optim_loss.backward()
        optim.step()

        train_loss.append(optim_loss.item())
        train_psnr.append(psnr.item())
        train_og_psnr.append(og_psnr.item())
        train_ssim.append(ssim.item())


        
    # run this part if test loader is defined
    if test_loader:
        model.eval()
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid, low_batch_valid in test_loader:

            high_batch_valid = high_batch_valid.to(device)
            low_batch_valid = low_batch_valid.to(device)

            with torch.no_grad():
                enhanced_batch_valid = model(low_batch_valid)
                # loss
                valid_loss_batch = torch.mean(
                    (high_batch_valid - enhanced_batch_valid)**2)
                # PSNR (dB) metric
                valid_psnr_batch = 20 * \
                    torch.log10(1. / torch.sqrt(valid_loss_batch))
                valid_ssim_batch = SSIM(enhanced_batch_valid, high_batch_valid)
            valid_psnr.append(valid_psnr_batch.item())
            valid_ssim.append(valid_ssim_batch.item())

    # We take the first image [0] from each batch
    high_img = high_batch_valid[0]
    low_img = low_batch_valid[0]
    enhanced_img = enhanced_batch_valid[0]

    caption = "1: Input, 2: Output, 3: Ground_Truth"
    images_list = [low_img, enhanced_img, high_img]
    images = log_images(images_list, caption)
    logger = {'train_loss': np.mean(train_loss), 'train_psnr': np.mean(train_psnr),
              'train_ssim': np.mean(train_ssim), 'train_og_psnr': np.mean(train_og_psnr), 
              'epoch': epoch,  'valid_psnr': np.mean(valid_psnr), 
              'valid_ssim': np.mean(valid_ssim), 'examples': images}

    if wandb_log:
        wandb.log(logger)


    print(f"Epoch {epoch + 1} of {last_epochs} took {time.time() - start_time:.3f}s\t Loss:{np.mean(train_loss)}\t PSNR:{np.mean(train_psnr)}\n")


    # Save the model after every epoch
    model_to_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': np.mean(train_loss),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(model_to_save, NEW_PATH_MODEL)

    #update scheduler
    scheduler.step()
    print('Scheduler last learning rate: ', scheduler.get_last_lr())
    

if wandb_log:
    wandb.finish()





