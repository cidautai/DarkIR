import torch
import wandb
import subprocess

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


if __name__ == '__main__':
    
    pass