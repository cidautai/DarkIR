import numpy as np
import os, sys
import time
import wandb
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
from lpips import LPIPS

from data.datasets.datapipeline import *
from archs import *
from losses import *
from data import *
from options.options import parse
from utils.utils import init_wandb


torch.autograd.set_detect_anomaly(True)

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need
path_options = './options/train/GOPRO.yml'
# print(os.path.isfile(path_options))
opt = parse(path_options)

# define some parameters based on the run we want to make
device = torch.device('cuda') if opt['device']['cuda'] else torch.device('cpu')

#parameters for saving model
PATH_MODEL     = opt['save']['path']
if opt['save']['new']:
    NEW_PATH_MODEL = opt['save']['new']
else: 
    NEW_PATH_MODEL = opt['save']['path']
    
BEST_PATH_MODEL = os.path.join(opt['save']['best'], os.path.basename(NEW_PATH_MODEL))

#---------------------------------------------------------------------------------------------------
# LOAD THE DATALOADERS
train_loader, test_loader = create_data(opt['datasets'])

#---------------------------------------------------------------------------------------------------
# DEFINE NETWORK, SCHEDULER AND OPTIMIZER
model, macs, params = create_model(opt['network'], cuda = opt['device']['cuda'])

# save this stats into opt to upload to wandb
opt['macs'] = macs
opt['params'] = params

model = freeze_parameters(model, substring='adapter', reverse = False) # freeze the adapter if there is any

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# define the optimizer
optim, scheduler = create_optim_scheduler(opt['train'], model)

# if resume load the weights
model, optim, scheduler, start_epochs = resume_model(model, optim, scheduler, path_model = PATH_MODEL, resume=opt['network']['resume_training'])

#---------------------------------------------------------------------------------------------------
# LOG INTO WANDB
init_wandb(opt)
#---------------------------------------------------------------------------------------------------
# DEFINE LOSSES AND METRICS
all_losses = create_loss(opt['train'])

calc_SSIM = SSIM(data_range=1.)
calc_LPIPS = LPIPS(net = 'vgg').to(device)

if opt['datasets']['train']['batch_size_train']>=8:
    largest_capable_size = opt['datasets']['train']['cropsize'] * opt['datasets']['train']['batch_size_train']
else: largest_capable_size = 1500

crop_to_4= CropTo4()
#---------------------------------------------------------------------------------------------------
# START THE TRAINING
best_valid_psnr = 0.
outside_batch = None

for epoch in tqdm(range(start_epochs, opt['train']['epochs'])):

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
        valid_lpips = []
    model.train()
    
    model = freeze_parameters(model, substring='adapter', reverse = False) # freeze the adapter if there is any
    optim_loss = 0

    for high_batch, low_batch in train_loader:

        # Move the data to the GPU
        high_batch = high_batch.to(device)
        low_batch = low_batch.to(device)

        optim.zero_grad()
        # Feed the data into the model
        enhanced_batch = model(low_batch)#, side_loss = False, use_adapter = None)
        # calculate loss function to optimize
        optim_loss = calculate_loss(all_losses, enhanced_batch, high_batch, outside_batch)

        # Calculate loss function for the PSNR
        loss = torch.mean((high_batch - enhanced_batch)**2)
        og_loss = torch.mean((high_batch - low_batch)**2)

        # and PSNR (dB) metric
        psnr = 20 * torch.log10(1. / torch.sqrt(loss))
        og_psnr = 20 * torch.log10(1. / torch.sqrt(og_loss))

        #calculate ssim metric
        ssim = calc_SSIM(enhanced_batch, high_batch)

        optim_loss.backward()
        optim.step()

        train_loss.append(optim_loss.item())
        train_psnr.append(psnr.item())
        train_og_psnr.append(og_psnr.item())
        train_ssim.append(ssim.item())


        
    # run this part if test loader is defined

    model.eval()
    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid, low_batch_valid in test_loader:

            _, _, H, W = high_batch_valid.shape
            
            if H >= largest_capable_size or W>=largest_capable_size: # we need to make crops
                high_batch_valid_crop, low_batch_valid_crop = crop_to_4(high_batch_valid, low_batch_valid) # this returns a list of crops of size [1, 3, H//2, W//2]

                valid_loss_batch = 0
                valid_ssim_batch = 0
                valid_lpips_batch = 0              
                for high_crop, low_crop in zip(high_batch_valid_crop, low_batch_valid_crop):
                    high_crop = high_crop.to(device)
                    low_crop = low_crop.to(device)

                    enhanced_crop = model(low_crop)#, use_adapter = None)
                    # loss
                    valid_loss_batch += torch.mean((high_crop - enhanced_crop)**2)
                    valid_ssim_batch += calc_SSIM(enhanced_crop, high_crop)
                    valid_lpips_batch += calc_LPIPS(enhanced_crop, high_crop)
                
                #the final value of lpips and ssim will be the mean of all the crops
                valid_ssim_batch  = valid_ssim_batch / 4
                valid_lpips_batch = valid_lpips_batch / 4
                enhanced_batch_valid = enhanced_crop
                high_batch_valid = high_crop
                low_batch_valid = low_crop
                
            else: # then we process the image normally
                high_batch_valid = high_batch_valid.to(device)
                low_batch_valid = low_batch_valid.to(device)
                enhanced_batch_valid = model(low_batch_valid)
                # loss
                valid_loss_batch = torch.mean((high_batch_valid - enhanced_batch_valid)**2)
                valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
                valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
                
            valid_psnr_batch = 20 * torch.log10(1. / torch.sqrt(valid_loss_batch))        
    
            valid_psnr.append(valid_psnr_batch.item())
            valid_ssim.append(valid_ssim_batch.item())
            valid_lpips.append(torch.mean(valid_lpips_batch).item())
            
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
              'valid_ssim': np.mean(valid_ssim), 'valid_lpips': np.mean(valid_lpips),'examples': images}

    if opt['wandb']['init']:
        wandb.log(logger)


    print(f"Epoch {epoch + 1} of {opt['train']['epochs']} took {time.time() - start_time:.3f}s\t Loss:{np.mean(train_loss)}\t PSNR:{np.mean(valid_psnr)}\n")

    weights = model.state_dict()
    baseline_weights = {k: v for k, v in weights.items() if 'adapter' not in k}

    # Save the model after every epoch
    model_to_save = {
        'epoch': epoch,
        'model_state_dict': baseline_weights,
        'optimizer_state_dict': optim.state_dict(),
        'loss': np.mean(train_loss),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(model_to_save, NEW_PATH_MODEL)

    #save best model if new valid_psnr is higher than the best one
    if np.mean(valid_psnr) >= best_valid_psnr:
        
        torch.save(model_to_save, BEST_PATH_MODEL)
        
        best_valid_psnr = np.mean(valid_psnr) # update best psnr

    #update scheduler
    scheduler.step()
    print('Scheduler last learning rate: ', scheduler.get_last_lr())
    
    
if opt['wandb']['init']:
    wandb.finish()





