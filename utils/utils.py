import wandb
import subprocess
from torchvision.utils import make_grid
import numpy as np

def init_wandb(opt):
    
    '''
    Initiates wandb if needed.
    opt: a dictionary from the yaml config 
    '''
    if opt['wandb']['init']:
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            project=opt['wandb']['project'], entity=opt['wandb']['entity'], 
            name=opt['wandb']['name'], save_code=opt['wandb']['save_code'],
            config = opt,
            resume = opt['wandb']['resume'],
            id = opt['wandb']['id'],
            notes= subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip() #latest github commit 
        )       
    else:
        print('Not uploading to wandb')

def create_grid(dict_images):
    '''
    A function to log the resulting images to wandb.
    images: A dictionary of images 
    '''
    images, caption = [], []
    for k, v in dict_images:
        caption.append(k)
        images.append(v)

    n=len(images)
    images_array = make_grid(images, n)
    
    images = wandb.Image(images_array, caption = caption)
    
    return images

def log_wandb(metrics, grid, log:bool = False):
    
    if log:
        logger = {'train_loss': np.mean(metrics['train_loss']), 'train_psnr': np.mean(metrics['train_psnr']),
                'train_ssim': np.mean(metrics['train_ssim']), 'train_og_psnr': np.mean(metrics['train_og_psnr']), 
                'epoch': metrics['epoch'],  'valid_psnr': np.mean(metrics['valid_psnr']), 
                'valid_ssim': np.mean(metrics['valid_ssim']), 'valid_lpips': np.mean(metrics['valid_lpips']),'examples': grid}
        wandb.log(logger)
    else:
        print('Not logging to wandb.')


if __name__ == '__main__':
    
    pass