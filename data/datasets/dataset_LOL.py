import os
from glob import glob

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *

def main_dataset_lol(rank = 0,train_path='/mnt/valab-datasets/LOL/train', test_path='/mnt/valab-datasets/LOL/test',
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512, flips = None,
                       num_workers=1, crop_type='Random', world_size=1):
    
    
    PATH_TRAIN = train_path
    PATH_VALID = test_path
    
    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, 'low', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'low'))]
    paths_sharp = [os.path.join(PATH_TRAIN, 'high', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'high'))]
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'low', path) for path in os.listdir(os.path.join(PATH_VALID, 'low'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'high', path) for path in os.listdir(os.path.join(PATH_VALID, 'high'))]        

    list_blur = paths_blur
    list_sharp = paths_sharp

    list_blur_valid = paths_blur_valid
    list_sharp_valid = paths_sharp_valid

    # check if all the image routes are correct
    check_paths([list_blur, list_blur_valid, list_sharp, list_sharp_valid])

    if verbose:
        print('Images in the subsets: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))

    tensor_transform = transforms.ToTensor()
    if flips:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
            transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
        ])
    else:
        flip_transform = None

    # Load the datasets
    train_dataset = MyDataset_Crop(list_blur, list_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    if world_size > 1:
        # Now we need to apply the Distributed sampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle= True, rank=rank)

        samplers = []
        # samplers = {'train': train_sampler, 'test': [test_sampler_gopro, test_sampler_lolblur]}
        samplers.append(train_sampler)
        samplers.append(test_sampler)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False)    
        samplers = None   

    return train_loader, test_loader, samplers

if __name__ == '__main__':
    
    train_loader, test_loader = main_dataset_lol(verbose = True, crop_type='Random', cropsize=256)
    print(len(train_loader), len(test_loader))
    
    for high, low in train_loader:
        
        print(high.shape)