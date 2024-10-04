import numpy as np
import os, sys
import time
import wandb
from tqdm import tqdm
from options.options import parse

path_options = './options/train/GOPRO.yml'
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt['device']['gpus'])

import torch
import torch.optim

from data.datasets.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import init_wandb, create_grid, log_wandb
from utils.train_utils import *

torch.autograd.set_detect_anomaly(True)

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need
# path_options = './options/train/GOPRO.yml'
# opt = parse(path_options)

# define some parameters based on the run we want to make
device = torch.device('cuda') if opt['device']['cuda'] else torch.device('cpu')

#parameters for saving model
PATH_MODEL     = opt['save']['path']
if opt['save']['new']:
    NEW_PATH_MODEL = opt['save']['new']
else: 
    NEW_PATH_MODEL = opt['save']['path']
    
BEST_PATH_MODEL = os.path.join(opt['save']['best'], os.path.basename(NEW_PATH_MODEL))

# LOAD THE DATALOADERS
train_loader, test_loader = create_data(opt['datasets'])

# DEFINE NETWORK, SCHEDULER AND OPTIMIZER
model, macs, params = create_model(opt['network'], cuda = opt['device']['cuda'])

# save this stats into opt to upload to wandb
opt['macs'] = macs
opt['params'] = params

model = freeze_parameters(model, substring='adapter', adapter = False) # freeze the adapter if there is any

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# define the optimizer
optim, scheduler = create_optim_scheduler(opt['train'], model)

# if resume load the weights
model, optim, scheduler, start_epochs = resume_model(model, optim, scheduler, path_model = PATH_MODEL, resume=opt['network']['resume_training'])

# INIT WANDB
init_wandb(opt)
# DEFINE LOSSES AND METRICS
all_losses = create_loss(opt['train'])

if opt['datasets']['train']['batch_size_train']>=8:
    largest_capable_size = opt['datasets']['train']['cropsize'] * opt['datasets']['train']['batch_size_train']
else: largest_capable_size = 1500

# START THE TRAINING
best_psnr = 0.

for epoch in tqdm(range(start_epochs, opt['train']['epochs'])):

    start_time = time.time()
    metrics = {'train_loss':[], 'train_psnr':[], 'train_og_psnr':[], 'train_ssim':[],
               'valid_psnr':[], 'valid_ssim':[], 'valid_lpips':[], 'epoch': epoch, 'best_psnr': best_psnr}
    # train phase
    model.train()
    model = freeze_parameters(model, substring='adapter', adapter = False) # freeze the adapter if there is any
    model, optim, metrics = train_model(model, optim, all_losses, train_loader, metrics)
    # eval phase
    model.eval()
    metrics, imgs_tensor = eval_model(model, test_loader, metrics, largest_capable_size=largest_capable_size)
    # log results to wandb       
    dict_images = { '1: Input': imgs_tensor['input'], '2: Output': imgs_tensor['output'], '3: Ground Truth': imgs_tensor['gt']}
    images_grid = create_grid(dict_images)
    log_wandb(metrics=metrics, grid = images_grid, log = opt['wandb']['init'])
    
    # print some results
    print(f"Epoch {epoch + 1} of {opt['train']['epochs']} took {time.time() - start_time:.3f}s\t Loss:{np.mean(metrics['train_loss'])}\t PSNR:{np.mean(metrics['valid_psnr'])}\n")
    # Save the model after every epoch
    metrics, best_psnr = save_checkpoint(model, optim, scheduler, metrics, paths = {'new':NEW_PATH_MODEL, 'best': BEST_PATH_MODEL})

    #update scheduler
    scheduler.step()
    print('Scheduler last learning rate: ', scheduler.get_last_lr())
    
if opt['wandb']['init']:
    wandb.finish()