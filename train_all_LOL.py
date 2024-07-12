import numpy as np
import os
import time
import wandb
from tqdm import tqdm
import subprocess

# PyTorch library
import torch
import torch.optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from ptflops import get_model_complexity_info

from data.datapipeline import *
from archs import Network
from archs import NAFNet
from losses.loss import MSELoss, L1Loss, CharbonnierLoss, SSIM, VGGLoss, EdgeLoss, FrequencyLoss

from data import *

from options.options import parse
from lpips import LPIPS
torch.autograd.set_detect_anomaly(True)

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need.
path_options = './options/train/All_LOL.yml'
print(os.path.isfile(path_options))
opt = parse(path_options)
# print(opt)
# define some parameters based on the run we want to make
# os.environ["CUDA_VISIBLE_DEVICES"]= '0, 1'
device = torch.device('cuda:0') if opt['device']['cuda'] else torch.device('cpu')

#selected network
network = opt['network']['name']

#parameters for saving model
PATH_MODEL     = opt['save']['path']
NEW_PATH_MODEL = opt['save']['path']
BEST_PATH_MODEL = os.path.join(opt['save']['best'], os.path.basename(opt['save']['path']))


wandb_log = opt['wandb']['init']  # flag if we want to output the results in wandb
resume_training = opt['resume_training']['resume_training'] # flag if we want to resume training

start_epochs = 0
last_epochs = opt['train']['epochs']

#---------------------------------------------------------------------------------------------------
# LOAD THE DATALOADERS
if opt['datasets']['name'] == 'All_LOL':
    train_loader, test_loader_real, test_loader_synth = main_dataset_all_lol(train_path=opt['datasets']['train']['train_path'],
                                                test_path = opt['datasets']['val']['test_path'],
                                                batch_size_train=opt['datasets']['train']['batch_size_train'],
                                                batch_size_test=opt['datasets']['val']['batch_size_test'],
                                                flips = opt['datasets']['train']['flips'],
                                                verbose=opt['datasets']['train']['verbose'],
                                                cropsize=opt['datasets']['train']['cropsize'],
                                                num_workers=opt['datasets']['train']['n_workers'],
                                                crop_type=opt['datasets']['train']['crop_type'])

else:
    name_loader = opt['datasets']['name']
    raise NotImplementedError(f'{name_loader} is not implemented')
#---------------------------------------------------------------------------------------------------
# DEFINE NETWORK, SCHEDULER AND OPTIMIZER

if network == 'Network':
    model = Network(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    dilations=opt['network']['dilations'],
                    extra_depth_wise=opt['network']['extra_depth_wise'])
elif network == 'NAFNet':
    model = NAFNet(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'])

else:
    raise NotImplementedError('This network isnt implemented')
model = model.to(device)

# if torch.cuda.device_count() > 1:
#     print("Usando", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)

#calculate MACs and number of parameters
macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat = False)
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
if opt['train']['lr_scheme'] == 'CosineAnnealing':
    scheduler = CosineAnnealingLR(optim, T_max=last_epochs, eta_min=opt['train']['eta_min'])
else: 
    raise NotImplementedError('scheduler not implemented')

# if resume load the weights
if resume_training:
    checkpoints = torch.load(PATH_MODEL)
    weights = checkpoints['model_state_dict']
    remove_prefix = 'module.' # this is needed because the keys now get a module. key that doesn't match with the network one
    weights = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in weights.items()}
    model.load_state_dict(weights)
    optim.load_state_dict(checkpoints['optimizer_state_dict']),
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    start_epochs = checkpoints['epoch']
    resume = opt['resume_training']['resume']
    id = opt['resume_training']['id']
    print('Loaded weights')
else:
    resume = 'never'
    id = None

#---------------------------------------------------------------------------------------------------
# LOG INTO WANDB
if wandb_log:
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project=opt['wandb']['project'], entity=opt['wandb']['entity'], 
        name=opt['wandb']['name'], save_code=opt['wandb']['save_code'],
        config = opt,
        resume = resume,
        id = id,
        notes= subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip() #latest github commit 
    )

#---------------------------------------------------------------------------------------------------
# DEFINE LOSSES AND METRICS

# first the pixel losses
if opt['train']['pixel_criterion'] == 'l1':
    pixel_loss = L1Loss()
elif opt['train']['pixel_criterion'] == 'l2':
    pixel_loss = MSELoss()
elif opt['train']['pixel_criterion'] == 'Charbonnier':
    pixel_loss = CharbonnierLoss()
else:
    raise NotImplementedError

# now the perceptual loss
perceptual = opt['train']['perceptual']
if perceptual:
    perceptual_loss = VGGLoss(loss_weight = opt['train']['perceptual_weight'],
                              criterion = opt['train']['perceptual_criterion'],
                              reduction = opt['train']['perceptual_reduction'])
else:
    perceptual_loss = None

#finally the edge loss
edge = opt['train']['edge'] 
if edge:
    edge_loss = EdgeLoss(loss_weight = opt['train']['edge_weight'],
                              criterion = opt['train']['edge_criterion'],
                              reduction = opt['train']['edge_reduction'])
else:
    edge_loss = None

# the frequency loss
frequency = opt['train']['frequency']
if frequency:
    frequency_loss = FrequencyLoss(loss_weight = opt['train']['edge_weight'],
                              reduction = opt['train']['edge_reduction'],
                              criterion = opt['train']['frequency_criterion'])

calc_SSIM = SSIM(data_range=1.)
calc_LPIPS = LPIPS(net = 'alex').to(device)

if opt['datasets']['train']['batch_size_train']>=8:
    largest_capable_size = opt['datasets']['train']['cropsize'] * opt['datasets']['train']['batch_size_train']
else: largest_capable_size = 1500

crop_to_4= CropTo4()
#---------------------------------------------------------------------------------------------------
# START THE TRAINING
best_valid_psnr = 0.

for epoch in tqdm(range(start_epochs, last_epochs)):

    start_time = time.time()
    train_loss = []
    train_psnr = []       # loss and PSNR of the enhanced-light image
    train_og_loss = []
    train_og_psnr = []  # loss and PSNR of the low-light image
    train_ssim    = []

    if test_loader_real:
        valid_loss_real = []
        valid_psnr_real = []
        valid_ssim_real = []
        valid_lpips_real = []
    if test_loader_synth:
        valid_loss_synth = []
        valid_psnr_synth = []
        valid_ssim_synth = []
        valid_lpips_synth = []
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
        if perceptual:
            l_pixel += perceptual_loss(enhanced_batch, high_batch)
        if edge:
            l_pixel += edge_loss(enhanced_batch, high_batch)
        if frequency:
            l_pixel += frequency_loss(enhanced_batch, high_batch)
        
        optim_loss = l_pixel

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
    if test_loader_real:
        model.eval()
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid_real, low_batch_valid_real in test_loader_real:

            _, _, H, W = high_batch_valid_real.shape
            if H >= largest_capable_size or W>=largest_capable_size:
                high_batch_valid_real, low_batch_valid_real = crop_to_4(high_batch_valid_real, low_batch_valid_real)
            
            high_batch_valid_real = high_batch_valid_real.to(device)
            low_batch_valid_real = low_batch_valid_real.to(device)

            with torch.no_grad():
                enhanced_batch_valid_real = model(low_batch_valid_real)
                # loss
                valid_loss_batch_real = torch.mean(
                    (high_batch_valid_real - enhanced_batch_valid_real)**2)
                # PSNR (dB) metric
                valid_psnr_batch_real = 20 * \
                    torch.log10(1. / torch.sqrt(valid_loss_batch_real))
                valid_ssim_batch_real = calc_SSIM(enhanced_batch_valid_real, high_batch_valid_real)
                valid_lpips_batch_real = calc_LPIPS(enhanced_batch_valid_real, high_batch_valid_real)
                
                
            valid_psnr_real.append(valid_psnr_batch_real.item())
            valid_ssim_real.append(valid_ssim_batch_real.item())
            valid_lpips_real.append(torch.mean(valid_lpips_batch_real).item())

    # We take the first image [0] from each batch real
    high_img_real = high_batch_valid_real[0]
    low_img_real = low_batch_valid_real[0]
    enhanced_img_real = enhanced_batch_valid_real[0]

    # run this part if test loader is defined
    if test_loader_synth:
        model.eval()
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for high_batch_valid_synth, low_batch_valid_synth in test_loader_synth:

            _, _, H, W = high_batch_valid_synth.shape
            if H >= largest_capable_size or W>=largest_capable_size:
                high_batch_valid_synth, low_batch_valid_synth = crop_to_4(high_batch_valid_synth, low_batch_valid_synth)
            
            high_batch_valid_synth = high_batch_valid_synth.to(device)
            low_batch_valid_synth = low_batch_valid_synth.to(device)

            with torch.no_grad():
                enhanced_batch_valid_synth = model(low_batch_valid_synth)
                # loss
                valid_loss_batch_synth = torch.mean(
                    (high_batch_valid_synth - enhanced_batch_valid_synth)**2)
                # PSNR (dB) metric
                valid_psnr_batch_synth = 20 * \
                    torch.log10(1. / torch.sqrt(valid_loss_batch_synth))
                valid_ssim_batch_synth = calc_SSIM(enhanced_batch_valid_synth, high_batch_valid_synth)
                valid_lpips_batch_synth = calc_LPIPS(enhanced_batch_valid_synth, high_batch_valid_synth)
                
                
            valid_psnr_synth.append(valid_psnr_batch_synth.item())
            valid_ssim_synth.append(valid_ssim_batch_synth.item())
            valid_lpips_synth.append(torch.mean(valid_lpips_batch_synth).item())
          
    # We take the first image [0] from each batch synth
    high_img_synth = high_batch_valid_synth[0]
    low_img_synth = low_batch_valid_synth[0]
    enhanced_img_synth = enhanced_batch_valid_synth[0]

    caption = "1: Input, 2: Output, 3: Ground_Truth"
    images_list_synth = [low_img_synth, enhanced_img_synth, high_img_synth]
    images_list_real = [low_img_real, enhanced_img_real, high_img_real]
    images_synth = log_images(images_list_synth, caption)
    images_real = log_images(images_list_real, caption)
    
    logger = {'train_loss': np.mean(train_loss), 'train_psnr': np.mean(train_psnr),
              'train_ssim': np.mean(train_ssim), 'train_og_psnr': np.mean(train_og_psnr), 
              'epoch': epoch,  'valid_psnr_real': np.mean(valid_psnr_real), 
              'valid_ssim_real': np.mean(valid_ssim_real), 'valid_lpips_real': np.mean(valid_lpips_real),
              'valid_psnr_synth': np.mean(valid_psnr_synth), 'valid_ssim_synth': np.mean(valid_ssim_synth), 
              'valid_lpips_synth': np.mean(valid_lpips_synth),'synth': images_synth, 'real': images_real}

    if wandb_log:
        wandb.log(logger)


    print(f"Epoch {epoch + 1} of {last_epochs} took {time.time() - start_time:.3f}s\t Loss:{np.mean(train_loss)}\t PSNR:{np.mean(valid_psnr_real)}\t PSNR:{np.mean(valid_psnr_synth)}\n")


    # Save the model after every epoch
    model_to_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': np.mean(train_loss),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(model_to_save, NEW_PATH_MODEL)

    #save best model if new valid_psnr is higher than the best one
    if np.mean(valid_psnr_real) >= best_valid_psnr:
        
        torch.save(model_to_save, BEST_PATH_MODEL)
        
        best_valid_psnr = np.mean(valid_psnr_real) # update best psnr

    #update scheduler
    scheduler.step()
    print('Scheduler last learning rate: ', scheduler.get_last_lr())
    
    

if wandb_log:
    wandb.finish()