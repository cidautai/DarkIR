import os
from glob import glob
import random

# PyTorch library
import torch
from torch.utils.data import DataLoader
import torch.optim

from .datapipeline import *
import cv2 as cv


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


def main_dataset_nbdn(train_path='/home/danfei/Python_Workspace/NBDN_dataset_50k/train', test_path='/home/danfei/Python_Workspace/NBDN_dataset_50k/test',
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512, flips = None,
                       num_workers=1, crop_type='Random'):

    PATH_TRAIN = train_path
    PATH_VALID = test_path

    # paths to the blur and sharp sets of images
    paths_blur = [os.path.join(PATH_TRAIN, 'blur', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'blur'))]
    paths_sharp = [os.path.join(PATH_TRAIN, 'sharp', path) for path in os.listdir(os.path.join(PATH_TRAIN, 'sharp'))]
    
    paths_blur_valid = [os.path.join(PATH_VALID, 'blur', path) for path in os.listdir(os.path.join(PATH_VALID, 'blur'))]
    paths_sharp_valid = [os.path.join(PATH_VALID, 'sharp', path) for path in os.listdir(os.path.join(PATH_VALID, 'sharp'))]    
    
    # extract the images from their corresponding folders, now we get a list of lists
    paths_blur = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in paths_blur ]
    paths_sharp = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element) * 100] for path_element in paths_sharp ]

    print(len(paths_blur), len(paths_sharp), len(paths_blur_valid), len(paths_sharp_valid))


    def flatten_list_comprehension(matrix):
        return [item for row in matrix for item in row]

    list_blur = flatten_list_comprehension(paths_blur)
    list_sharp = flatten_list_comprehension(paths_sharp)

    list_blur_valid = paths_blur_valid
    list_sharp_valid = paths_sharp_valid

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur +
             list_sharp+list_blur_valid+list_sharp_valid]
    for true in trues:
        if true != True:
            print('Non valid route!')

    print('Images in the subsets: \n')
    print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur))
    print("    -Images in the PATH_HIGH_TRAINING folder: ", len(list_sharp))
    print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
    print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))

    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
    tensor_transform = transforms.ToTensor()
    if flips:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
            transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
        ])

    # Load the datasets
    train_dataset = MyDataset_Crop(list_blur, list_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=cropsize,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    # #Load the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    # #test_loader = None

    return train_loader, test_loader

if __name__ == '__main__':
    train, test = main_dataset_nbdn(train_path='../../../NBDN_dataset_50k/train', test_path='../../../NBDN_dataset_50k/test',
                       batch_size_train=8, batch_size_test=1, verbose=True, cropsize=256,
                       num_workers=1, crop_type='Center') 
    print(len(train), len(test))