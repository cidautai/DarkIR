import numpy as np
import os
import time
import wandb
from tqdm import tqdm

# PyTorch library
import torch
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ptflops import get_model_complexity_info

from data.datapipeline import *
from archs import Network
from archs import NAFNet
from archs.network_v2 import Network as Network_v2
from archs.network_v3 import Network as Network_v3
from archs.network_MBNv4 import Network as Network_MBNv4
from losses.loss import MSELoss, L1Loss, CharbonnierLoss, SSIMloss, SSIM
from data.dataset_NBDN import main_dataset_nbdn
from data.dataset_LOLBlur import main_dataset_lolblur
from options.options import parse
from lpips import LPIPS


# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need.
path_options = '/home/danfei/Python_workspace/deblur/Net-Low-Light-Deblurring/options/train/LOLBlur_MBNv4.yml'
print(os.path.isfile(path_options))
opt = parse(path_options)
# print(opt)
# define some parameters based on the run we want to make
device = torch.device('cuda')

#selected network
network = opt['network']['name']

#parameters for saving model
PATH_MODEL     = opt['save']['path']
NEW_PATH_MODEL = opt['save']['path']
BEST_PATH_MODEL = os.path.join(opt['save']['best'], os.path.basename(opt['save']['path']))

#---------------------------------------------------------------

wandb_log = opt['wandb']['init']  # flag if we want to output the results in wandb

resume_training = opt['resume_training']['resume_training'] # flag if we want to resume training

start_epochs = 0
last_epochs = opt['train']['epochs']


#load the dataloaders
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
if opt['datasets']['name'] == 'LOLBlur':
    train_loader, test_loader = main_dataset_lolblur(train_path=opt['datasets']['train']['train_path'],
                                                test_path = opt['datasets']['val']['test_path'],
                                                batch_size_train=opt['datasets']['train']['batch_size_train'],
                                                batch_size_test=opt['datasets']['val']['batch_size_test'],
                                                flips = opt['datasets']['train']['flips'],
                                                verbose=opt['datasets']['train']['verbose'],
                                                cropsize=opt['datasets']['train']['cropsize'],
                                                num_workers=opt['datasets']['train']['n_workers'],
                                                crop_type=opt['datasets']['train']['crop_type'])

if network == 'Network':
    model = Network(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    dilations=opt['network']['dilations'])
elif network == 'NAFNet':
    model = NAFNet(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
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

if network == 'Network_MBNv4':
    model = Network_MBNv4(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    expand_ratio=opt['network']['expand_ratio'])

else:
    raise NotImplementedError
model = model.to(device)

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
    resume = opt['resume_training']['resume']
    id = opt['resume_training']['id']
else:
    resume = 'never'
    id = None

# log into wandb
if wandb_log:
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project=opt['wandb']['project'], entity=opt['wandb']['entity'], 
        name=opt['wandb']['name'], save_code=opt['wandb']['save_code'],
        config = opt,
        resume = resume,
        id = id
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

calc_SSIM = SSIM(data_range=1.)
calc_LPIPS = LPIPS(net = 'vgg').to(device)


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
                valid_ssim_batch = calc_SSIM(enhanced_batch_valid, high_batch_valid)
                valid_lpips_batch = calc_LPIPS(enhanced_batch_valid, high_batch_valid)
                
                
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





