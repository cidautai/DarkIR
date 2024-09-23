import numpy as np
import os, sys
import time
import wandb
from tqdm import tqdm

# PyTorch library
import torch
import torch.optim
from ptflops import get_model_complexity_info

from data.datapipeline import *
from archs import Network
from archs import NAFNet
from losses.loss import SSIM
from data.dataset_LOLBlur import main_dataset_lolblur
from data.dataset_LOL import main_dataset_lol
from data import main_dataset_gopro, main_dataset_lolv2, main_dataset_lolv2_synth
from options.options import parse
from lpips import LPIPS
from utils.utils import load_weights

path_options = './options/train/Finetune.yml'


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
PATH_MODEL     = opt['save']['path_model']
PATH_ADAPTER   = opt['save']['path']
# NEW_PATH_MODEL = opt['save']['path']
# BEST_PATH_MODEL = os.path.join(opt['save']['best'], os.path.basename(opt['save']['path']))


wandb_log = opt['wandb']['init']  # flag if we want to output the results in wandb
resume_training = opt['resume_training']['resume_training'] # flag if we want to resume training

start_epochs = 0
last_epochs = opt['train']['epochs']

#---------------------------------------------------------------------------------------------------
# LOAD THE DATALOADERS
if opt['datasets']['name'] == 'GOPRO':
    train_loader, test_loader = main_dataset_gopro(train_path=opt['datasets']['train']['train_path'],
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
                    middle_blk_num_enc=opt['network']['middle_blk_num_enc'],
                    middle_blk_num_dec=opt['network']['middle_blk_num_dec'], 
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

#calculate MACs and number of parameters
macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat = False)
print('Computational complexity: ', macs)
print('Number of parameters: ', params)

# save this stats into opt to upload to wandb
opt['macs'] = macs
opt['params'] = params

# define the optimizer
# optim = torch.optim.AdamW(model.parameters(), lr = opt['train']['lr_initial'],
#                           weight_decay = opt['train']['weight_decay'],
#                           betas = opt['train']['betas'])

# Initialize the cosine annealing scheduler
# if opt['train']['lr_scheme'] == 'CosineAnnealing':
#     scheduler = CosineAnnealingLR(optim, T_max=last_epochs, eta_min=opt['train']['eta_min'])
# else: 
#     raise NotImplementedError('scheduler not implemented')

# if resume load the weights
checkpoints_model = torch.load(PATH_MODEL)
checkpoints_adapter = torch.load(PATH_ADAPTER)

# first load the weights of the baseline model
model = load_weights(model, old_weights = checkpoints_model['model_state_dict'])
# Then, load the weights of the adapter
# model = load_weights(model, old_weights=checkpoints_adapter['model_state_dict'])
# print(checkpoints_model['model_state_dict'].keys())

#---------------------------------------------------------------------------------------------------
# DEFINE METRICS

# sys.exit()

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
        enhanced_batch_valid = model(low_batch_valid, adapter=False)
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