import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from ptflops import get_model_complexity_info

from .nafnet_utils.arch_model import NAFNet
from .network import Network
from .arch_util import load_weights, load_optim, freeze_parameters

def create_model(opt, cuda):
    '''
    Creates the model.
    opt: a dictionary from the yaml config key network
    '''
    
    name = opt['name']

    device = torch.device('cuda') if cuda else torch.device('cpu') 
    if name == 'Network':
        model = Network(img_channel=opt['img_channels'], 
                        width=opt['width'], 
                        middle_blk_num_enc=opt['middle_blk_num_enc'],
                        middle_blk_num_dec=opt['middle_blk_num_dec'], 
                        enc_blk_nums=opt['enc_blk_nums'],
                        dec_blk_nums=opt['dec_blk_nums'], 
                        dilations=opt['dilations'],
                        extra_depth_wise=opt['extra_depth_wise'])
    elif name == 'NAFNet':
        model = NAFNet(img_channel=opt['img_channels'], 
                        width=opt['width'], 
                        middle_blk_num=opt['middle_blk_num_enc'], 
                        enc_blk_nums=opt['enc_blk_nums'],
                        dec_blk_nums=opt['dec_blk_nums'])

    else:
        raise NotImplementedError('This network is not implemented')
    print(f'Using {name} network')
    
    model.to(device)
    input_size = (3, 256, 256)
    macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
    print(f'Computational complexity at {input_size}: {macs}')
    print('Number of parameters: ', params)    

    return model, macs, params

def create_optim_scheduler(opt, model):
    '''
    Returns the optim and its scheduler.
    opt: a dictionary of the yaml config file with the train key
    '''
    optim = torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()) , 
                            lr = opt['lr_initial'],
                            weight_decay = opt['weight_decay'],
                            betas = opt['betas'])
    
    if opt['lr_scheme'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optim, T_max=opt['epochs'], eta_min=opt['eta_min'])
    else: 
        raise NotImplementedError('scheduler not implemented')    
        
    return optim, scheduler

def resume_model(model,
                 optim,
                 scheduler, 
                 path_model, resume:str=None):
    
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    if resume:
        checkpoints = torch.load(path_model)
        weights = checkpoints['model_state_dict']
        model = load_weights(model, old_weights=weights)
        optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'], model = model)
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        start_epochs = checkpoints['epoch']

        print('Loaded weights')
    else:
        start_epochs = 0
        print('Starting from zero the training')
    
    return model, optim, scheduler, start_epochs



__all__ = ['create_model', 'resume_model', 'freeze_parameters', 'create_optim_scheduler']



    