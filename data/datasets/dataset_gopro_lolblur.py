import os
from glob import glob
import random

# PyTorch library
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim

import cv2 as cv

try:
    from .datapipeline import *
except:
    from datapipeline import *

def create_path(IMGS_PATH, list_new_files):
    '''
    Util function to add the file path of all the images to the list of names of the selected 
    images that will form the valid ones.
    '''
    file_path, name = os.path.split(
        IMGS_PATH[0])  # we pick only one element of the list
    output = [os.path.join(file_path, element) for element in list_new_files]

    return output


def common_member(a, b):
    '''
    Returns true if the two lists (valid and training) have a common element.
    '''
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False


def random_sort_pairs(list1, list2):
    '''
    This function makes the same random sort to each list, so that they are sorted and the pairs are maintained.
    '''
    # Combine the lists
    combined = list(zip(list1, list2))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip back into separate lists
    list1[:], list2[:] = zip(*combined)

    return list1, list2

def flatten_list_comprehension(matrix):
    return [item for row in matrix for item in row]

def main_dataset_gopro_lolblur(rank=1, train_path='/mnt/valab-datasets/GOPRO/train', test_path='/mnt/valab-datasets/GOPRO/test',
                       batch_size_train=4, flips = None, batch_size_test=1, verbose=False, cropsize=512,
                       num_workers=1, crop_type='Random', world_size = 1):

    # begin by loading the gopro dataset
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

    def flatten_list_comprehension(matrix):
        return [item for row in matrix for item in row]

    # we scale the number of gopro images to the images in lolblur
    list_blur_gopro = flatten_list_comprehension(list_blur)
    list_sharp_gopro = flatten_list_comprehension(list_sharp)

    list_blur_valid_gopro = flatten_list_comprehension(list_blur_valid)
    list_sharp_valid_gopro = flatten_list_comprehension(list_sharp_valid)

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur_gopro +
             list_sharp_gopro+list_blur_valid_gopro+list_sharp_valid_gopro]
    for true in trues:
        if true != True:
            print('Non valid route!')

    print('Images in the GOPRO subsets: \n')
    print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur_gopro))
    print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp_gopro))
    print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_gopro))
    print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_gopro))
    
    print(len(list_blur_gopro))
    # --------------------------------------------------------------------------------------
    # now load the lolblur dataset

    PATH_TRAIN = '/home/danfei/Datasets/LOLBlur/train_HARD'
    PATH_VALID = '/home/danfei/Datasets/LOLBlur/test'
    
    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, 'low_blur_noise', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'low_blur_noise'))]
    paths_sharp = [os.path.join(PATH_TRAIN, 'high_sharp_scaled', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'high_sharp_scaled'))]
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'low_blur_noise', path) for path in os.listdir(os.path.join(PATH_VALID, 'low_blur_noise'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'high_sharp_scaled', path) for path in os.listdir(os.path.join(PATH_VALID, 'high_sharp_scaled'))]        
    
    # print(len(paths_blur), len(paths_blur_valid), len(paths_sharp), len(paths_sharp_valid))
    
    # extract the images from their corresponding folders, now we get a list of lists
    # paths_blur = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_blur ]
    # paths_sharp = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_sharp ]

    paths_blur_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_blur_valid ]
    paths_sharp_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_sharp_valid ]

    list_blur_lolblur = paths_blur
    list_sharp_lolblur = paths_sharp

    list_blur_valid_lolblur = flatten_list_comprehension(paths_blur_valid)
    list_sharp_valid_lolblur = flatten_list_comprehension(paths_sharp_valid)

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur_lolblur +
             list_sharp_lolblur+list_blur_valid_lolblur+list_sharp_valid_lolblur]
    for true in trues:
        if true != True:
            print('Non valid route!')

    if verbose:
        print('Images in the LOLBlur subsets: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur_lolblur))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp_lolblur))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_lolblur))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_lolblur))


    list_blur  = list_blur_gopro + list_blur_lolblur
    list_sharp = list_sharp_gopro + list_sharp_lolblur
    # list_blur_valid = list_blur_valid_gopro + list_blur_valid_lolblur
    # list_sharp_valid = list_sharp_valid_gopro + list_sharp_valid_lolblur 

    # if verbose:
    #     print('Total images in the subsets: \n')
    #     print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur))
    #     print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp))
    #     print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
    #     print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))

    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
    tensor_transform = transforms.ToTensor()
    if flips:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
            transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
        ])
    else: flip_transform= None

    # Load the datasets
    train_dataset = MyDataset_Crop(list_blur, list_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)
    test_dataset_gopro = MyDataset_Crop(list_blur_valid_gopro, list_sharp_valid_gopro, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)
    test_dataset_lolblur = MyDataset_Crop(list_blur_valid_lolblur, list_sharp_valid_lolblur, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    if world_size > 1:
        # Now we need to apply the Distributed sampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, rank=rank)
        test_sampler_gopro = DistributedSampler(test_dataset_gopro, num_replicas=world_size, shuffle= True, rank=rank)
        test_sampler_lolblur = DistributedSampler(test_dataset_lolblur, num_replicas=world_size, shuffle= True, rank=rank)

        samplers = []
        # samplers = {'train': train_sampler, 'test': [test_sampler_gopro, test_sampler_lolblur]}
        samplers.append(train_sampler)
        samplers.append(test_sampler_gopro)
        samplers.append(test_sampler_lolblur)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=True, sampler = train_sampler)
        test_loader_gopro = DataLoader(dataset=test_dataset_gopro, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler = test_sampler_gopro)
        test_loader_lolblur = DataLoader(dataset=test_dataset_lolblur, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler= test_sampler_lolblur)
        
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=True, sampler = None)
        test_loader_gopro = DataLoader(dataset=test_dataset_gopro, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler = None)
        test_loader_lolblur = DataLoader(dataset=test_dataset_lolblur, batch_size=batch_size_test, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler= None)       
        samplers = None

    test_loader = {'gopro':test_loader_gopro, 'lolblur':test_loader_lolblur}

    return train_loader, test_loader, samplers

if __name__ == '__main__':
    
    train_loader, test_loader_gopro, test_loader_lolblur = main_dataset_gopro_lolblur(verbose = True, crop_type='Random', cropsize=256)
    print(len(train_loader), len(test_loader_gopro), len(test_loader_lolblur))
    
    for high, low in train_loader:
        
        print(high.shape)