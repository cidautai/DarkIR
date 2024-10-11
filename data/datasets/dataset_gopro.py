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

def main_dataset_gopro(rank = 1,
                       train_path='../../GOPRO_dataset/train', 
                       test_path='../../GOPRO_dataset/test',
                       batch_size_train=4, 
                       batch_size_test=1, 
                       verbose=False, 
                       cropsize=512,
                       flips = None,
                       num_workers=1, 
                       crop_type='Random',
                       world_size = 1):

    PATH_TRAIN = train_path
    PATH_VALID = test_path

    dirs_train = os.listdir(PATH_TRAIN)
    dirs_valid = os.listdir(PATH_VALID)
    # print(dirs_train)
    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, x, 'blur') for x in dirs_train]
    paths_sharp = [os.path.join(PATH_TRAIN, x, 'sharp') for x in dirs_train]

    paths_blur_valid = [os.path.join(PATH_VALID, x, 'blur')
                        for x in dirs_valid]
    paths_sharp_valid = [os.path.join(
        PATH_VALID, x, 'sharp') for x in dirs_valid]

    # add the blurred and sharp images from each gopro
    list_blur = [glob(os.path.join(x, '*.png')) for x in paths_blur]
    list_sharp = [glob(os.path.join(x, '*.png')) for x in paths_sharp]

    list_blur_valid = [glob(os.path.join(x, '*.png'))
                       for x in paths_blur_valid]
    list_sharp_valid = [glob(os.path.join(x, '*.png'))
                        for x in paths_sharp_valid]

    list_blur = flatten_list_comprehension(list_blur)#[:200]
    list_sharp = flatten_list_comprehension(list_sharp)#[:200]

    list_blur_valid = flatten_list_comprehension(list_blur_valid)#[:30]
    list_sharp_valid = flatten_list_comprehension(list_sharp_valid)#[:30]

    # check if all the image routes are correct
    check_paths([list_blur, list_blur_valid, list_sharp, list_sharp_valid])

    if verbose:
        print('Images in the subsets:')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))

    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
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
                                   tensor_transform=tensor_transform, flips=flip_transform, 
                                   test=False, crop_type=crop_type)
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, 
                                  crop_type=crop_type)

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
    
    PATH_TRAIN = '/mnt/valab-datasets/LOLBlur/train'
    PATH_VALID = '/mnt/valab-datasets/LOLBlur/test'

    dirs_train = os.listdir(PATH_TRAIN)
    dirs_valid = os.listdir(PATH_VALID)
    # print(dirs_train)
    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, x, 'blur') for x in dirs_train]
    paths_sharp = [os.path.join(PATH_TRAIN, x, 'sharp') for x in dirs_train]

    paths_blur_valid = [os.path.join(PATH_VALID, x, 'blur')
                        for x in dirs_valid]
    paths_sharp_valid = [os.path.join(
        PATH_VALID, x, 'sharp') for x in dirs_valid]

    # add the blurred and sharp images from each gopro
    list_blur = [glob(os.path.join(x, '*.png')) for x in paths_blur]
    list_sharp = [glob(os.path.join(x, '*.png')) for x in paths_sharp]