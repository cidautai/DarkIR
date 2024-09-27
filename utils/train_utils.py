import torch
import sys
from lpips import LPIPS

sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')
from losses import *
from data.datasets.datapipeline import CropTo4

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
calc_SSIM = SSIM(data_range=1.)
calc_LPIPS = LPIPS(net = 'vgg', verbose=False).to(device)
crop_to_4= CropTo4()

def train_model(model, optim, all_losses, train_loader, metrics):
    '''
    It trains the model, returning the model, optim, scheduler and metrics dict
    '''
    outside_batch = None
    for high_batch, low_batch in train_loader:

        # Move the data to the GPU
        high_batch = high_batch.to(device)
        low_batch = low_batch.to(device)

        optim.zero_grad()
        # Feed the data into the model
        enhanced_batch = model(low_batch)#, side_loss = False, use_adapter = None)
        # print(enhanced_batch)
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
        # print(optim_loss)
        optim_loss.requires_grad = True
        optim_loss.backward()
        optim.step()

        metrics['train_loss'].append(optim_loss.item())
        metrics['train_psnr'].append(psnr.item())
        metrics['train_og_psnr'].append(og_psnr.item())
        metrics['train_ssim'].append(ssim.item())    
    
    return model, optim, metrics

def eval_model(model, test_loader, metrics, largest_capable_size = 1500):
    
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
    
            metrics['valid_psnr'].append(valid_psnr_batch.item())
            metrics['valid_ssim'].append(valid_ssim_batch.item())
            metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())
    imgs_tensor = {'input':low_batch_valid[0], 'output':enhanced_batch_valid[0], 'gt':high_batch_valid[0]}
    return metrics, imgs_tensor