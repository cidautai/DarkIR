from .datasets.dataset_gopro import main_dataset_gopro
from .datasets.dataset_LOL import main_dataset_lol
from .datasets.dataset_LOLBlur import main_dataset_lolblur
from .datasets.dataset_LOLv2 import main_dataset_lolv2, main_dataset_lolv2_synth
# from .dataset_all_LOL import main_dataset_all_lol
from .datasets.dataset_gopro_lolblur import main_dataset_gopro_lolblur
# from .create_data import create_data

def create_data(rank, world_size, opt):
    '''
    opt: a dictionary from the yaml config key datasets 
    '''
    name = opt['name']
    train_path=opt['train']['train_path']
    test_path = opt['val']['test_path']
    batch_size_train=opt['train']['batch_size_train']
    batch_size_test=opt['val']['batch_size_test']
    flips = opt['train']['flips']
    verbose=opt['train']['verbose']
    cropsize=opt['train']['cropsize']
    num_workers=opt['train']['n_workers']
    crop_type=opt['train']['crop_type']  
    
    if rank != 0:
        verbose = False
    samplers = None # TEmporal change!!
    if name == 'LOLBlur':
        train_loader, test_loader, samplers = main_dataset_lolblur(rank = rank,
                                                train_path=train_path,
                                                test_path = test_path,
                                                batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test,
                                                flips = flips,
                                                verbose=verbose,
                                                cropsize=cropsize,
                                                num_workers=num_workers,
                                                crop_type=crop_type,
                                                world_size = world_size)
    elif name == 'LOL':
        train_loader, test_loader, samplers = main_dataset_lol( rank = rank,
                                                train_path=train_path,
                                                test_path = test_path,
                                                batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test,
                                                flips = flips,
                                                verbose=verbose,
                                                cropsize=cropsize,
                                                num_workers=num_workers,
                                                crop_type=crop_type,
                                                world_size = world_size )   
    elif name == 'LOLv2':
        train_loader, test_loader, samplers = main_dataset_lolv2( rank = rank,
                                                train_path=train_path,
                                                test_path = test_path,
                                                batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test,
                                                flips = flips,
                                                verbose=verbose,
                                                cropsize=cropsize,
                                                num_workers=num_workers,
                                                crop_type=crop_type,
                                                world_size=world_size)   
    elif name == 'LOLv2_synth':
        train_loader, test_loader, samplers = main_dataset_lolv2_synth(rank=rank, 
                                                train_path=train_path,
                                                test_path = test_path,
                                                batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test,
                                                flips = flips,
                                                verbose=verbose,
                                                cropsize=cropsize,
                                                num_workers=num_workers,
                                                crop_type=crop_type, 
                                                world_size=world_size)   

    elif name == 'GOPRO':
        train_loader, test_loader, samplers = main_dataset_gopro( rank=rank,
                                                train_path=train_path,
                                                test_path = test_path,
                                                batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test,
                                                flips = flips,
                                                verbose=verbose,
                                                cropsize=cropsize,
                                                num_workers=num_workers,
                                                crop_type=crop_type,
                                                world_size=world_size)
    elif name == 'GOPRO_LOLBlur':
        train_loader, test_loader, samplers = main_dataset_gopro_lolblur( rank=rank,
                                                train_path=train_path,
                                                test_path = test_path,
                                                batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test,
                                                flips = flips,
                                                verbose=verbose,
                                                cropsize=cropsize,
                                                num_workers=num_workers,
                                                crop_type=crop_type,
                                                world_size=world_size)  

    else:
        raise NotImplementedError(f'{name} is not implemented')        
    # print(samplers, train_loader, test_loader)
    print(f'Using {name} Dataset')
    
    return train_loader, test_loader, samplers

__all__ = ['create_data']