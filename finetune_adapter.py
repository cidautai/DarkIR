import numpy as np
import os, sys
import time
import wandb
from tqdm import tqdm
from options.options import parse

# read the options file and define the variables from it. If you want to change the hyperparameters of the net and the conditions of training go to
# the file and change them what you need
path_options = './options/train/GOPRO.yml'
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt['device']['gpus']) # you need to fix this before importing torch

import torch
import torch.optim
import torch.multiprocessing as mp

from data.datasets.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import init_wandb, logging_dict, create_path_models
from utils.train_utils import *
torch.autograd.set_detect_anomaly(True)

# define some parameters based on the run we want to make
device = torch.device('cuda') if opt['device']['cuda'] else torch.device('cpu')

#parameters for saving model
PATH_MODEL, PATH_ADAPTER, NEW_PATH_ADAPTER, BEST_PATH_ADAPTER = create_path_models(opt['save'])

if opt['datasets']['train']['batch_size_train']>=8:
    largest_capable_size = opt['datasets']['train']['cropsize'] * opt['datasets']['train']['batch_size_train']
else: largest_capable_size = 1500

best_psnr = 0.
    
def run_model(rank, world_size):
    
    setup(rank, world_size=world_size)

    # LOAD THE DATALOADERS
    train_loader, test_loader, samplers = create_data(rank, world_size=world_size, opt = opt['datasets'])
    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, macs, params = create_model(opt['network'], cuda = opt['device'], rank=rank, adapter=True)

    # save this stats into opt to upload to wandb
    opt['macs'] = macs
    opt['params'] = params

    # define the optimizer
    optim, scheduler = create_optim_scheduler(opt['train'], model)

    # if resume load the weights
    model, optim, scheduler, start_epochs = resume_adapter(model, optim, scheduler, path_adapter = PATH_ADAPTER, 
                                                        path_model = PATH_MODEL, rank = rank, resume=opt['network']['resume_training'])

    all_losses = create_loss(opt['train'], rank=rank)
    # INIT WANDB
    init_wandb(rank, opt)
    best_psnr= 0
    for epoch in tqdm(range(start_epochs, opt['train']['epochs'])):

        start_time = time.time()
        metrics_train = {'epoch': epoch,'best_psnr': best_psnr}
        metrics_eval = {}

        # shuffle the samplers of each loader
        shuffle_sampler(samplers, epoch)
        # train phase
        model.train()
        # model = freeze_parameters(model, substring='adapter', adapter = False) # freeze the adapter if there is any
        model, optim, metrics_train = train_model(model, optim, all_losses, train_loader,
                                            metrics_train, adapter = True, rank = rank)
        # eval phase
        model.eval()
        metrics_eval, imgs_dict = eval_model(model, test_loader, metrics_eval, 
                                                    largest_capable_size=largest_capable_size, adapter= True, rank=rank)
        
        # print some results
        print(f"Epoch {epoch + 1} of {opt['train']['epochs']} took {time.time() - start_time:.3f}s\n")
        if type(next(iter(metrics_eval.values()))) == dict:
            for key, metric_eval in metrics_eval.items():
                print(f' \t {key} --- PSNR: {metric_eval['valid_psnr']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}')
        else:
            print(f' \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}')
        # Save the model after every epoch
        best_psnr = save_checkpoint(model, optim, scheduler, metrics_eval = metrics_eval, metrics_train=metrics_train, 
                                    paths = {'new':NEW_PATH_ADAPTER, 'best': BEST_PATH_ADAPTER}, rank=rank)

        # log into wandb if needed
        if opt['wandb']['init'] and rank == 0: wandb.log(logging_dict(metrics_train, metrics_eval, imgs_dict))
        #update scheduler
        scheduler.step()

    cleanup()

def main():
    # world_size = len(opt['device']['ids'])
    world_size = len(opt['device']['ids'])
    mp.spawn(run_model, args =(world_size,), nprocs=world_size, join=True)

    if opt['wandb']['init']:
        wandb.finish()

if __name__ == '__main__':
    main()



