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
from archs import NAFNet, NAFNet_dilated_124, NAFNet_dilated_124_concat, NAFNet_dilated_14, NAFNet_dilated_14_concat
from losses.loss import MSELoss, PerceptualLoss, L1Loss
from data.dataset import main_dataset_gopro, main_dataset_gopro_realblur, main_dataset_realblur_gopro
from data.dataset_NBDN import main_dataset_nbdn

from torch.optim.lr_scheduler import CosineAnnealingLR

# define some parameters based on the run we want to make
device = torch.device('cuda')
batch_size_train = 8
batch_size_test = 1
PATH_MODEL = 'models/NAFNet_base_Added_L1_cosine_annealing_RealBlurandGOPRO.pt'
NEW_PATH_MODEL = 'models/NAFNet_base_Added_L1_cosine_annealing_RealBlurandGOPRO.pt'

wandb_log = True  # flag if we want to output the results in wandb
resume_training = False  # flag if we want to resume training
test_loader = None

start_epochs = 0
last_epochs = 1500

# branch = block_branch_try(3, 2, FFN_Expand = 2, dilation = 1, drop_out_rate = 0)
# print('Defining model')
model = NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[2, 2, 4, 8],
               dec_blk_nums=[2, 2, 2, 2])

print('Number of parameters: ', sum(p.numel()
      for p in model.parameters() if p.requires_grad))
model = model.to(device)
# and its optimizer
optim = torch.optim.AdamW(model.parameters(), lr=1e-4,
                          weight_decay=1e-3, betas=[0.9, 0.9])

if resume_training:
    checkpoints = torch.load(PATH_MODEL)
    model.load_state_dict(checkpoints['model_state_dict'])
    optim.load_state_dict(checkpoints['optimizer_state_dict'])
    start_epochs = checkpoints['epoch']


if wandb_log:

    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="motion-stabilization", entity="cidautai", name="NAFNet_base_Added_L1_cosine_annealing_RealBlurandGOPRO", save_code=True
    )

# Initialize the cosine annealing scheduler
# we want the cycle of iterations to
T_max = 200 // batch_size_train * last_epochs
# be the same as the total number of iterations
# print(T_max)
scheduler = CosineAnnealingLR(optim, T_max=T_max, eta_min=1e-6)

train_path = '../../GOPRO_dataset/train'
test_path = '../../GOPRO_dataset/test'

train_loader, test_loader = main_dataset_realblur_gopro(train_path, test_path, batch_size_train=batch_size_train,
                                               batch_size_test=batch_size_test, verbose=True, cropsize=256,
                                               num_workers=4, crop_type='Center')

# we define the losses
pixel_loss = L1Loss()

for epoch in tqdm(range(start_epochs, last_epochs)):

    start_time = time.time()
    train_loss = []
    train_psnr = []       # loss and PSNR of the normal-light image
    train_og_loss = []
    train_og_psnr = []  # loss and PSNR of the low-light image

    if test_loader:
        valid_loss = []
        valid_psnr = []
    model.train()
    optim_loss = 0
    # i = 0
    for high_batch, low_batch in train_loader:

        # Move the data to the GPU
        high_batch = high_batch.to(device)
        low_batch = low_batch.to(device)

        optim.zero_grad()

        # Feed the data into the model
        enhanced_batch = model(low_batch)

        # print(enhanced_batch.device, high_batch.device)
        # calculate loss function to optimize
        l_pixel = pixel_loss(enhanced_batch, high_batch)
        # l_percep, l_style = perceptual_loss(enhanced_batch, high_batch)
        optim_loss = l_pixel  # + l_percep

        # Calculate loss function for the PSNR
        loss = torch.mean((high_batch - enhanced_batch)**2)
        og_loss = torch.mean((high_batch - low_batch)**2)

        # and PSNR (dB) metric
        psnr = 20 * torch.log10(1. / torch.sqrt(loss))
        og_psnr = 20 * torch.log10(1. / torch.sqrt(og_loss))

        optim_loss.backward()
        optim.step()

        train_loss.append(optim_loss.item())
        train_psnr.append(psnr.item())
        # train_og_loss.append(og_loss.item())
        train_og_psnr.append(og_psnr.item())
        # i+=1
        # print(f'{i} Batch completed')

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

            # valid_loss.append(valid_loss_batch.item())
            valid_psnr.append(valid_psnr_batch.item())

    # We take the first image [0] from each batch, and convert it to numpy array
    high_img = high_batch[0]
    low_img = low_batch[0]
    enhanced_img = enhanced_batch[0]

    caption = "1: Input, 2: Output, 3: Ground_Truth"
    images_list = [low_img, enhanced_img, high_img]
    images = log_images(images_list, caption)
    logger = {'train_loss': np.mean(train_loss), 'train_psnr': np.mean(train_psnr),
              'train_og_psnr': np.mean(train_og_psnr), 'epoch': epoch,  'valid_psnr': np.mean(valid_psnr), 'examples': images}

    if wandb_log:
        wandb.log(logger)
        # log_images(images_list, epoch, caption)

    print(f"Epoch {epoch + 1} of {last_epochs} took {time.time() - start_time:.3f}s\t Loss:{np.mean(train_loss)}\t PSNR:{np.mean(train_psnr)}\n")

    # Save the model after every epoch
    model_to_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': np.mean(train_loss)
    }
    torch.save(model_to_save, NEW_PATH_MODEL)
    scheduler.step()
    # train_loader, test_loader = main_dataset(batch_size_train=batch_size_train, batch_size_test= batch_size_test, verbose = False)

if wandb_log:
    wandb.finish()

# #----------------------SECOND TRAINING
# # define some parameters based on the run we want to make
# device = torch.device('cuda')
# batch_size_train = 4
# batch_size_test = 2
# PATH_MODEL = 'models/NAFNet_base16_encoder3_L1_cosine_annealing_GOPRO200.pt'
# NEW_PATH_MODEL = 'models/NAFNet_base16_encoder3_L1_cosine_annealing_GOPRO200.pt'

# wandb_log = True  # flag if we want to output the results in wandb
# resume_training = False  # flag if we want to resume training
# test_loader = None

# start_epochs = 0
# last_epochs = 500

# # branch = block_branch_try(3, 2, FFN_Expand = 2, dilation = 1, drop_out_rate = 0)
# # print('Defining model')
# model = NAFNet(img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[2, 2, 4],
#                dec_blk_nums=[2, 2, 2])

# print('Number of parameters: ', sum(p.numel()
#       for p in model.parameters() if p.requires_grad))
# model = model.to(device)
# # and its optimizer
# optim = torch.optim.AdamW(model.parameters(), lr=1e-4,
#                           weight_decay=1e-3, betas=[0.9, 0.9])

# if resume_training:
#     checkpoints = torch.load(PATH_MODEL)
#     model.load_state_dict(checkpoints['model_state_dict'])
#     optim.load_state_dict(checkpoints['optimizer_state_dict'])
#     start_epochs = checkpoints['epoch']


# if wandb_log:

#     wandb.login()
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="motion-stabilization", entity="cidautai", name="NAFNet_base16_encoder3_L1_cosine_annealing_GOPRO200", save_code=True
#     )

# # Initialize the cosine annealing scheduler
# # we want the cycle of iterations to
# T_max = 200 // batch_size_train * last_epochs
# # be the same as the total number of iterations
# # print(T_max)
# scheduler = CosineAnnealingLR(optim, T_max=T_max, eta_min=1e-6)

# train_path = '../../GOPRO_dataset/train'
# test_path = '../../GOPRO_dataset/test'

# train_loader, test_loader = main_dataset_gopro(train_path, test_path, batch_size_train=batch_size_train,
#                                                batch_size_test=batch_size_test, verbose=True, cropsize=256,
#                                                num_workers=4, crop_type='Center')

# # we define the losses
# pixel_loss = L1Loss()
# # pixel_loss = MSELoss()
# # perceptual_loss = PerceptualLoss(layer_weights={'conv5_4': 1},
# #                                  vgg_type='vgg19',
# #                                  use_input_norm=True,
# #                                  perceptual_weight=1e-2,
# #                                  style_weight=0.,
# #                                  norm_img=False,
# #                                  criterion='l1')


# for epoch in tqdm(range(start_epochs, last_epochs)):

#     start_time = time.time()
#     train_loss = []
#     train_psnr = []       # loss and PSNR of the normal-light image
#     train_og_loss = []
#     train_og_psnr = []  # loss and PSNR of the low-light image

#     if test_loader:
#         valid_loss = []
#         valid_psnr = []
#     model.train()
#     optim_loss = 0
#     # i = 0
#     for high_batch, low_batch in train_loader:

#         # Move the data to the GPU
#         high_batch = high_batch.to(device)
#         low_batch = low_batch.to(device)

#         optim.zero_grad()

#         # Feed the data into the model
#         enhanced_batch = model(low_batch)

#         # print(enhanced_batch.device, high_batch.device)
#         # calculate loss function to optimize
#         l_pixel = pixel_loss(enhanced_batch, high_batch)
#         # l_percep, l_style = perceptual_loss(enhanced_batch, high_batch)
#         optim_loss = l_pixel  # + l_percep

#         # Calculate loss function for the PSNR
#         loss = torch.mean((high_batch - enhanced_batch)**2)
#         og_loss = torch.mean((high_batch - low_batch)**2)

#         # and PSNR (dB) metric
#         psnr = 20 * torch.log10(1. / torch.sqrt(loss))
#         og_psnr = 20 * torch.log10(1. / torch.sqrt(og_loss))

#         optim_loss.backward()
#         optim.step()

#         train_loss.append(optim_loss.item())
#         train_psnr.append(psnr.item())
#         # train_og_loss.append(og_loss.item())
#         train_og_psnr.append(og_psnr.item())
#         # i+=1
#         # print(f'{i} Batch completed')

#     # run this part if test loader is defined
#     if test_loader:
#         model.eval()
#         # Now we need to go over the test_loader and evaluate the results of the epoch
#         for high_batch_valid, low_batch_valid in test_loader:

#             high_batch_valid = high_batch_valid.to(device)
#             low_batch_valid = low_batch_valid.to(device)

#             with torch.no_grad():
#                 enhanced_batch_valid = model(low_batch_valid)
#                 # loss
#                 valid_loss_batch = torch.mean(
#                     (high_batch_valid - enhanced_batch_valid)**2)
#                 # PSNR (dB) metric
#                 valid_psnr_batch = 20 * \
#                     torch.log10(1. / torch.sqrt(valid_loss_batch))

#             # valid_loss.append(valid_loss_batch.item())
#             valid_psnr.append(valid_psnr_batch.item())

#     # We take the first image [0] from each batch, and convert it to numpy array
#     high_img = high_batch[0]
#     low_img = low_batch[0]
#     enhanced_img = enhanced_batch[0]

#     caption = "1: Input, 2: Output, 3: Ground_Truth"
#     images_list = [low_img, enhanced_img, high_img]
#     images = log_images(images_list, caption)
#     logger = {'train_loss': np.mean(train_loss), 'train_psnr': np.mean(train_psnr),
#               'train_og_psnr': np.mean(train_og_psnr), 'epoch': epoch,  'valid_psnr': np.mean(valid_psnr), 'examples': images}

#     if wandb_log:
#         wandb.log(logger)
#         # log_images(images_list, epoch, caption)

#     print(f"Epoch {epoch + 1} of {last_epochs} took {time.time() - start_time:.3f}s\t Loss:{np.mean(train_loss)}\t PSNR:{np.mean(train_psnr)}\n")

#     # Save the model after every epoch
#     model_to_save = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optim.state_dict(),
#         'loss': np.mean(train_loss)
#     }
#     torch.save(model_to_save, NEW_PATH_MODEL)
#     scheduler.step()
#     # train_loader, test_loader = main_dataset(batch_size_train=batch_size_train, batch_size_test= batch_size_test, verbose = False)

# if wandb_log:
#     wandb.finish()







