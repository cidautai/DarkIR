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
from losses.loss import MSELoss, L1Loss, CharbonnierLoss, SSIM, VGGLoss, EdgeLoss
from data.dataset_NBDN import main_dataset_nbdn
from data.dataset_LOLBlur import main_dataset_lolblur
from options.options import parse
from lpips import LPIPS

path_options = './options/train/LOLBlur.yml'


# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need.
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
                    dilations=opt['network']['dilations'],
                    extra_depth_wise=opt['network']['extra_depth_wise'])

elif network == 'Network_MBNv4':
    model = Network_MBNv4(img_channel=opt['network']['img_channels'], 
                    width=opt['network']['width'], 
                    middle_blk_num=opt['network']['middle_blk_num'], 
                    enc_blk_nums=opt['network']['enc_blk_nums'],
                    dec_blk_nums=opt['network']['dec_blk_nums'], 
                    residual_layers=opt['network']['residual_layers'],
                    expand_ratio=opt['network']['expand_ratio'])

else:
    raise NotImplementedError('This network isnt implemented')
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
if opt['train']['lr_scheme'] == 'CosineAnnealing':
    scheduler = CosineAnnealingLR(optim, T_max=last_epochs, eta_min=opt['train']['eta_min'])
else: 
    raise NotImplementedError('scheduler not implemented')

# if resume load the weights
if resume_training:
    checkpoints = torch.load(BEST_PATH_MODEL)
    model.load_state_dict(checkpoints['model_state_dict'])
    optim.load_state_dict(checkpoints['optimizer_state_dict']),
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    start_epochs = checkpoints['epoch']
    resume = opt['resume_training']['resume']
    id = opt['resume_training']['id']
else:
    resume = 'never'
    id = None


#---------------------------------------------------------------------------------------------------
# DEFINE METRICS

calc_SSIM = SSIM(data_range=1.)
calc_LPIPS = LPIPS(net = 'vgg').to(device)


valid_psnr = []
valid_ssim = []
valid_lpips = []

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

print(f'PSNR validation value: {np.mean(valid_psnr)}')
print(f'SSIM validation value: {np.mean(valid_ssim)}')
print(f'LPIPS validation value: {np.mean(valid_lpips)}')