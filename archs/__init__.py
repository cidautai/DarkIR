import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from ptflops import get_model_complexity_info

from .nafnet_utils.arch_model import NAFNet
from .network import Network
from .arch_util import load_weights, load_optim, save_checkpoint

def freeze_parameters(model,
                      substring:str,
                      adapter:bool = False):
    if adapter:
        for name, param in model.named_parameters():
            if substring not in name:
                param.requires_grad = False  
        return model
    else:
        for name, param in model.named_parameters():
            if substring in name:
                param.requires_grad = False  
        return model        

    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}") 

def create_model(opt, cuda,rank, adapter = False, substring = 'adapter'):
    '''
    Creates the model.
    opt: a dictionary from the yaml config key network
    '''
    print(rank)
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
                        dec_blk_nums=opt['dec_blk_nums'])#.to(rank)

    else:
        raise NotImplementedError('This network is not implemented')
    print(f'Using {name} network')

    input_size = (3, 256, 256)
    macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
    print(f'Computational complexity at {input_size}: {macs}')
    print('Number of parameters: ', params)    
    # if torch.cuda.device_count() > 1:
    #     print("Usando", torch.cuda.device_count(), "GPUs!")
    #     model = DataParallel(model)
    #if wanted, distribute into different gpus
    # print(cuda['ids'])
    # if len(cuda['ids']) > 1:
    #     # model.to(rank)
    #     model = DDP(model, device_ids=[rank])
    # else:
    #     model.to(device)
    model.to(rank)
    
    #freeze the parameters before feeding into the DDP
    model = freeze_parameters(model, substring=substring, adapter=adapter)
    model = DDP(model, device_ids=[rank])
    
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
                 path_model, 
                 rank,resume:str=None):
    
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    map_location = {'hip:%d' % 0: 'hip:%d' % rank}
    if resume:
        checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
        weights = checkpoints['model_state_dict']
        model = load_weights(model, old_weights=weights)
        optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        start_epochs = checkpoints['epoch']

        print('Loaded weights')
    else:
        start_epochs = 0
        print('Starting from zero the training')
    
    return model, optim, scheduler, start_epochs

def resume_adapter(model,
                 optim,
                 scheduler, 
                 path_adapter,
                 path_model,
                 rank, resume:str=None):
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    #first load the model weights
    checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
    weights = checkpoints['model_state_dict']
    model = load_weights(model, old_weights=weights) 
    #now if needed load the adapter weights
    if resume:
        checkpoints = torch.load(path_adapter,map_location=map_location, weights_only=False)
        weights = checkpoints['model_state_dict']
        model = load_weights(model, old_weights=weights)
        # optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        start_epochs = checkpoints['epoch']

        print('Loaded weights')
    else:
        start_epochs = 0
        print('Starting from zero the training')
    
    return model, optim, scheduler, start_epochs


__all__ = ['create_model', 'resume_model', 'resume_adapter', 'freeze_parameters', 'create_optim_scheduler', 'save_checkpoint']



    