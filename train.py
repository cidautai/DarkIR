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
path_options = './options/train/LOLBlur.yml'
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
if opt['save']['new']:
    NEW_PATH_MODEL = opt['save']['new']
else: 
    NEW_PATH_MODEL = opt['save']['path']
    
BEST_PATH_MODEL = os.path.join(opt['save']['best'], os.path.basename(NEW_PATH_MODEL))


wandb_log = opt['wandb']['init']  # flag if we want to output the results in wandb
resume_training = opt['resume_training']['resume_training'] # flag if we want to resume training

start_epochs = 0
last_epochs = opt['train']['epochs']

#---------------------------------------------------------------------------------------------------
# LOAD THE DATALOADERS
if opt['datasets']['name'] == 'NBDN':
    train_loader, test_loader = main_dataset_nbdn(train_path=opt['datasets']['train']['train_path'],
                                                test_path = opt['datasets']['val']['test_path'],
                                                batch_size_train=opt['datasets']['train']['batch_size_train'],
                                                batch_size_test=opt['datasets']['val']['batch_size_test'],
                                                flips = opt['datasets']['train']['flips'],
                                                verbose=opt['datasets']['train']['verbose'],
                                                cropsize=opt['datasets']['train']['cropsize'],
                                                num_workers=opt['datasets']['train']['n_workers'],
                                                crop_type=opt['datasets']['train']['crop_type'])
elif opt['datasets']['name'] == 'LOLBlur':
    train_loader, test_loader = main_dataset_lolblur(train_path=opt['datasets']['train']['train_path'],
                                                test_path = opt['datasets']['val']['test_path'],
                                                batch_size_train=opt['datasets']['train']['batch_size_train'],
                                                batch_size_test=opt['datasets']['val']['batch_size_test'],
                                                flips = opt['datasets']['train']['flips'],
                                                verbose=opt['datasets']['train']['verbose'],
                                                cropsize=opt['datasets']['train']['cropsize'],
                                                num_workers=opt['datasets']['train']['n_workers'],
                                                crop_type=opt['datasets']['train']['crop_type'])
elif opt['datasets']['name'] == 'LOL':
    train_loader, test_loader = main_dataset_lol(train_path=opt['datasets']['train']['train_path'],
                                                test_path = opt['datasets']['val']['test_path'],
                                                batch_size_train=opt['datasets']['train']['batch_size_train'],
                                                batch_size_test=opt['datasets']['val']['batch_size_test'],
                                                flips = opt['datasets']['train']['flips'],
                                                verbose=opt['datasets']['train']['verbose'],
                                                cropsize=opt['datasets']['train']['cropsize'],
                                                num_workers=opt['datasets']['train']['n_workers'],
                                                crop_type=opt['datasets']['train']['crop_type'])
elif opt['datasets']['name'] == 'LOLv2':
    train_loader, test_loader = main_dataset_lolv2(train_path=opt['datasets']['train']['train_path'],
                                                test_path = opt['datasets']['val']['test_path'],
                                                batch_size_train=opt['datasets']['train']['batch_size_train'],
                                                batch_size_test=opt['datasets']['val']['batch_size_test'],
                                                flips = opt['datasets']['train']['flips'],
                                                verbose=opt['datasets']['train']['verbose'],
                                                cropsize=opt['datasets']['train']['cropsize'],
                                                num_workers=opt['datasets']['train']['n_workers'],
                                                crop_type=opt['datasets']['train']['crop_type'])    
elif opt['datasets']['name'] == 'LOLv2_synth':
    train_loader, test_loader = main_dataset_lolv2_synth(train_path=opt['datasets']['train']['train_path'],
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
    # print(weights.keys())
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

# the edge loss
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
calc_LPIPS = LPIPS(net = 'vgg').to(device)

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

    if test_loader:
        valid_loss = []
        valid_psnr = []
        valid_ssim = []
        valid_lpips = []
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

                    enhanced_crop = model(low_crop)
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

    if wandb_log:
        wandb.log(logger)


    print(f"Epoch {epoch + 1} of {last_epochs} took {time.time() - start_time:.3f}s\t Loss:{np.mean(train_loss)}\t PSNR:{np.mean(valid_psnr)}\n")


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
    if np.mean(valid_psnr) >= best_valid_psnr:
        
        torch.save(model_to_save, BEST_PATH_MODEL)
        
        best_valid_psnr = np.mean(valid_psnr) # update best psnr

    #update scheduler
    scheduler.step()
    print('Scheduler last learning rate: ', scheduler.get_last_lr())
    
    

if wandb_log:
    wandb.finish()





