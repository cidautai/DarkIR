import os
from glob import glob
import random

# PyTorch library
import torch
from torch.utils.data import DataLoader
import torch.optim

import cv2 as cv

try:
    from .datapipeline import *
except:
    from data.datasets.datapipeline import *
    
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

def main_dataset_all_lol(train_path='/home/leadergpu/Datasets', test_path='/home/leadergpu/Datasets',
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512, flips = None,
                       num_workers=1, crop_type='Random'):
    
    #first lets load the LOL dataset list
    PATH_TRAIN = os.path.join(train_path, 'LOL', 'train')
    PATH_VALID = os.path.join(test_path, 'LOL', 'test')
    
    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, 'low', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'low'))]
    paths_sharp = [os.path.join(PATH_TRAIN, 'high', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'high'))]
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'low', path) for path in os.listdir(os.path.join(PATH_VALID, 'low'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'high', path) for path in os.listdir(os.path.join(PATH_VALID, 'high'))]        
    
    # print(paths_blur)
    # print(len(paths_blur), len(paths_blur_valid), len(paths_sharp), len(paths_sharp_valid))
    

    list_blur_LOL = paths_blur
    list_sharp_LOL = paths_sharp

    list_blur_valid_LOL = paths_blur_valid
    list_sharp_valid_LOL = paths_sharp_valid

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur_LOL +
             list_sharp_LOL+list_blur_valid_LOL+list_sharp_valid_LOL]
    for true in trues:
        if true != True:
            print('Non valid route!')

    if verbose:
        print('Images in the subsets of LOL: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur_LOL))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp_LOL))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_LOL))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_LOL), '\n')

    #------------------------------------------------------------------------
    # now load the LOLv2_real dataset
    
    PATH_TRAIN = os.path.join(train_path, 'LOL-v2/Real_captured', 'Train')
    PATH_VALID = os.path.join(test_path, 'LOL-v2/Real_captured', 'Test')

    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, 'Low', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'Low'))]
    paths_sharp = [os.path.join(PATH_TRAIN, 'Normal', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'Normal'))]
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'Low', path) for path in os.listdir(os.path.join(PATH_VALID, 'Low'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'Normal', path) for path in os.listdir(os.path.join(PATH_VALID, 'Normal'))]        
    
    # print(paths_blur)
    # print(len(paths_blur), len(paths_blur_valid), len(paths_sharp), len(paths_sharp_valid))
    list_blur_LOLv2_real = paths_blur
    list_sharp_LOLv2_real = paths_sharp

    list_blur_valid_LOLv2_real = paths_blur_valid
    list_sharp_valid_LOLv2_real = paths_sharp_valid

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur_LOLv2_real +
             list_sharp_LOLv2_real+list_blur_valid_LOLv2_real+list_sharp_valid_LOLv2_real]
    for true in trues:
        if true != True:
            print('Non valid route!')

    if verbose:
        print('Images in the subsets of LOLv2-real: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur_LOLv2_real))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp_LOLv2_real))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_LOLv2_real))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_LOLv2_real), '\n')

    #------------------------------------------------------------------------
    # now load the LOLv2_synth dataset    
    PATH_TRAIN = os.path.join(train_path, 'LOL-v2/Synthetic', 'Train')
    PATH_VALID = os.path.join(test_path, 'LOL-v2/Synthetic', 'Test')
    
    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, 'Low', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'Low'))]
    paths_sharp = [os.path.join(PATH_TRAIN, 'Normal', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'Normal'))]
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'Low', path) for path in os.listdir(os.path.join(PATH_VALID, 'Low'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'Normal', path) for path in os.listdir(os.path.join(PATH_VALID, 'Normal'))]        
    
    # print(paths_blur)
    print(len(paths_blur), len(paths_blur_valid), len(paths_sharp), len(paths_sharp_valid))

    list_blur_LOLv2_synth = paths_blur
    list_sharp_LOLv2_synth = paths_sharp

    list_blur_valid_LOLv2_synth = paths_blur_valid
    list_sharp_valid_LOLv2_synth = paths_sharp_valid

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur_LOLv2_synth +
             list_sharp_LOLv2_synth+list_blur_valid_LOLv2_synth+list_sharp_valid_LOLv2_synth]
    for true in trues:
        if true != True:
            print('Non valid route!')

    if verbose:
        print('Images in the subsets of LOLv2-synth: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur_LOLv2_synth))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp_LOLv2_synth))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_LOLv2_synth))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_LOLv2_synth), '\n')    

    #------------------------------------------------------------------------  
    # finally the LOLBlur dataset
    PATH_TRAIN = os.path.join(train_path, 'LOLBlur_temp', 'train')
    PATH_VALID = os.path.join(test_path, 'LOLBlur_temp', 'test')
    
    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, 'low_blur_noise', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'low_blur_noise'))]
    paths_sharp = [os.path.join(PATH_TRAIN, 'high_sharp_scaled', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'high_sharp_scaled'))]
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'low_blur_noise', path) for path in os.listdir(os.path.join(PATH_VALID, 'low_blur_noise'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'high_sharp_scaled', path) for path in os.listdir(os.path.join(PATH_VALID, 'high_sharp_scaled'))]        
    
    # extract the images from their corresponding folders, now we get a list of lists
    paths_blur = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_blur ]
    paths_sharp = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_sharp ]

    paths_blur_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_blur_valid ]
    paths_sharp_valid = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_sharp_valid ]



    list_blur_lolblur = flatten_list_comprehension(paths_blur)
    list_sharp_lolblur = flatten_list_comprehension(paths_sharp)

    list_blur_valid_lolblur = flatten_list_comprehension(paths_blur_valid)
    list_sharp_valid_lolblur = flatten_list_comprehension(paths_sharp_valid)

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur_lolblur +
             list_sharp_lolblur+list_blur_valid_lolblur+list_sharp_valid_lolblur]
    for true in trues:
        if true != True:
            print('Non valid route!')

    if verbose:
        print('Images in the subsets of LOL-Blur: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur_lolblur))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp_lolblur))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_lolblur))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_lolblur), '\n')

    #------------------------------------------------------------------------  
    # now add all the regular lol datasets  
    list_blur_lol  = list_blur_LOL + list_blur_LOLv2_real + list_blur_LOLv2_synth 
    list_sharp_lol = list_sharp_LOL + list_sharp_LOLv2_real + list_sharp_LOLv2_synth
    
    list_blur_valid_lol  = list_blur_valid_LOL + list_blur_valid_LOLv2_real + list_blur_valid_LOLv2_synth
    list_sharp_valid_lol = list_sharp_valid_LOL + list_sharp_valid_LOLv2_real +list_sharp_valid_LOLv2_synth

    if verbose:
        print('Total images in the subsets of lol: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur_lol * 5))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp_lol * 5))
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid_lol))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid_lol), '\n')

    #------------------------------------------------------------------------  
    # finally add the lol and lolblur, augmenting the lol datasets to a ratio 1:1 with lolblur
    list_blur = list_blur_lol * 5 + list_blur_lolblur
    list_sharp = list_sharp_lol * 5 + list_sharp_lolblur

    if verbose:
        print('Total images for training in the subsets!: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp))


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
    #the test sets will be especific to lolv2 real and synth
    test_dataset_lolv2 = MyDataset_Crop(list_blur_valid_LOLv2_real, list_sharp_valid_LOLv2_real, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    test_dataset_lolv2_synth = MyDataset_Crop(list_blur_valid_LOLv2_synth, list_sharp_valid_LOLv2_synth, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)
    
    test_dataset_lolblur = MyDataset_Crop(list_blur_valid_lolblur, list_sharp_valid_lolblur, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)
    # print(list_blur_valid_LOLv2_real[:10], '\n',list_blur_valid_LOLv2_synth[:10])
    # #Load the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader_lolv2 = DataLoader(dataset=test_dataset_lolv2, batch_size=batch_size_test, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader_lolv2_synth = DataLoader(dataset=test_dataset_lolv2_synth, batch_size=batch_size_test, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader_lolblur = DataLoader(dataset=test_dataset_lolblur, batch_size=batch_size_test, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
  
    # #test_loader = None

    return train_loader, test_loader_lolv2, test_loader_lolv2_synth, test_loader_lolblur

if __name__ == '__main__':
    
    train_loader, test_loader_real, test_loader_synth, test_loader_lolblur = main_dataset_all_lol(verbose = True, crop_type='Random', cropsize=256)
    print(len(train_loader), len(test_loader_real), len(test_loader_synth), len(test_loader_lolblur))
    
    # for high, low in train_loader:
    #     # pass
    #     print(high.shape)