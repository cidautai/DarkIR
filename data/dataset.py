import os
from glob import glob
import random

# PyTorch library
import torch
from torch.utils.data import DataLoader
import torch.optim


import cv2 as cv

from .datapipeline import *

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


def main_dataset_gopro(train_path='../../GOPRO_dataset/train', test_path='../../GOPRO_dataset/test',
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512,
                       num_workers=1, crop_type='Random'):

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

    list_blur = flatten_list_comprehension(list_blur)#[:200]
    list_sharp = flatten_list_comprehension(list_sharp)#[:200]

    list_blur_valid = flatten_list_comprehension(list_blur_valid)#[:30]
    list_sharp_valid = flatten_list_comprehension(list_sharp_valid)#[:30]

    # we random sort the lists using random_sort_pairs
    
    list_blur, list_sharp             = random_sort_pairs(list_blur, list_sharp)
    list_blur_valid, list_sharp_valid = random_sort_pairs(list_blur_valid, list_sharp_valid)

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur +
             list_sharp+list_blur_valid+list_sharp_valid]
    for true in trues:
        if true != True:
            print('Non valid route!')

    image_blur = cv.imread(list_blur[0])
    image_sharp_valid = cv.imread(list_sharp_valid[0])
    print(image_blur.shape, image_sharp_valid.shape)

    # list_blur  = list_blur + list_blur_valid[:1010]
    # list_sharp = list_sharp + list_sharp_valid[:1010]

    # list_blur_valid  = list_blur_valid[1011:]
    # list_sharp_valid = list_sharp_valid[1011:]

    print('Images in the subsets: \n')
    print("    -Images in the PATH_LOW_TRAINING folder: ", len(list_blur))
    print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))

    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
    tensor_transform = transforms.ToTensor()
    flip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
        transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
    ])

    # Load the datasets
    train_dataset = MyDataset_Crop(list_blur, list_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=cropsize,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    # # print(len(train_dataset))
    # # print(train_dataset[0])
    # # for high, low in train_dataset:
    # #     print(type(high))
    # #Load the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    # #test_loader = None

    return train_loader, test_loader

def main_dataset_gopro_realblur(train_path='../../GOPRO_dataset/train', test_path='../../GOPRO_dataset/test',
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512,
                       num_workers=1, crop_type='Random'):

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

    list_blur = flatten_list_comprehension(list_blur)#[:200]
    list_sharp = flatten_list_comprehension(list_sharp)#[:200]

    list_blur_valid = flatten_list_comprehension(list_blur_valid)#[:30]
    list_sharp_valid = flatten_list_comprehension(list_sharp_valid)#[:30]

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur +
             list_sharp+list_blur_valid+list_sharp_valid]
    for true in trues:
        if true != True:
            print('Non valid route in GOPRO!')



    #Now we read the images of RealBlur and append their paths to another lists

    path_txt = '/home/danfei/Python_Workspace/RealBlur'
    blur_list, sharp_list = [], []
    blur_list_valid, sharp_list_valid = [], []

    with open(os.path.join(path_txt, 'RealBlur_J_train_list.txt')) as file:
        for line in file:
            sharp, blur = line.strip().split()
            blur_list.append(blur)
            sharp_list.append(sharp)

    with open(os.path.join(path_txt, 'RealBlur_J_test_list.txt')) as file:
        for line in file:
            sharp, blur = line.strip().split()
            blur_list_valid.append(blur)
            sharp_list_valid.append(sharp)



    #print(len(blur_list), len(sharp_list), len(sharp_list_valid), len(blur_list_valid))

    blur_list  = [os.path.join(path_txt, path) for path in blur_list]
    sharp_list = [os.path.join(path_txt, path) for path in sharp_list]

    blur_list_valid  = [os.path.join(path_txt, path) for path in blur_list_valid]
    sharp_list_valid = [os.path.join(path_txt, path) for path in sharp_list_valid]

    trues = [os.path.isfile(file) for file in blur_list +
                sharp_list + blur_list_valid + sharp_list_valid]

    for true in trues:
        if true != True:
            print('Non valid route in RBlur!')

    # Print the number of images of the datasets
    print('Images in the subsets: \n')
    print("    -Images in the GOPRO_train folder: ", len(list_blur), len(list_sharp))
    print("    -Images in the GOPRO_test folder : ", len(list_blur_valid), len(list_sharp_valid))
    print("    -Images in the RBlur_train folder: ", len(blur_list), len(sharp_list))
    print("    -Images in the RBlur_test folder : ", len(blur_list_valid), len(sharp_list_valid))



    # Now we fuse the two datasets: For the evaluation we only use GOPRO
    list_blur  = list_blur + blur_list
    list_sharp = list_sharp + sharp_list
    
    list_blur_valid  = list_blur_valid
    list_sharp_valid = list_sharp_valid 

    print('Total number of images:')
    print('     - Train:', len(list_blur), len(list_sharp))
    print('     - Test: ', len(list_blur_valid), len(list_sharp_valid))

    # we random sort the lists using random_sort_pairs
    
    list_blur, list_sharp             = random_sort_pairs(list_blur, list_sharp)
    list_blur_valid, list_sharp_valid = random_sort_pairs(list_blur_valid, list_sharp_valid)

    image_blur = cv.imread(list_blur[0])
    image_sharp_valid = cv.imread(list_sharp[0])
    print(image_blur.shape, image_sharp_valid.shape)



    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
    tensor_transform = transforms.ToTensor()
    flip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
        transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
    ])

    # Load the datasets
    train_dataset = MyDataset_Crop(list_blur, list_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=cropsize,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    # # print(len(train_dataset))
    # # print(train_dataset[0])
    # # for high, low in train_dataset:
    # #     print(type(high))
    # #Load the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    # #test_loader = None

    return train_loader, test_loader

def main_dataset_realblur_gopro(train_path='../../GOPRO_dataset/train', test_path='../../GOPRO_dataset/test',
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512,
                       num_workers=1, crop_type='Random'):

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

    list_blur = flatten_list_comprehension(list_blur)#[:200]
    list_sharp = flatten_list_comprehension(list_sharp)#[:200]

    list_blur_valid = flatten_list_comprehension(list_blur_valid)#[:30]
    list_sharp_valid = flatten_list_comprehension(list_sharp_valid)#[:30]

    # check if all the image routes are correct
    trues = [os.path.isfile(file) for file in list_blur +
             list_sharp+list_blur_valid+list_sharp_valid]
    for true in trues:
        if true != True:
            print('Non valid route in GOPRO!')



    #Now we read the images of RealBlur and append their paths to another lists

    path_txt = '/home/danfei/Python_Workspace/RealBlur'
    blur_list, sharp_list = [], []
    blur_list_valid, sharp_list_valid = [], []

    with open(os.path.join(path_txt, 'RealBlur_J_train_list.txt')) as file:
        for line in file:
            sharp, blur = line.strip().split()
            blur_list.append(blur)
            sharp_list.append(sharp)

    with open(os.path.join(path_txt, 'RealBlur_J_test_list.txt')) as file:
        for line in file:
            sharp, blur = line.strip().split()
            blur_list_valid.append(blur)
            sharp_list_valid.append(sharp)



    #print(len(blur_list), len(sharp_list), len(sharp_list_valid), len(blur_list_valid))

    blur_list  = [os.path.join(path_txt, path) for path in blur_list]
    sharp_list = [os.path.join(path_txt, path) for path in sharp_list]

    blur_list_valid  = [os.path.join(path_txt, path) for path in blur_list_valid]
    sharp_list_valid = [os.path.join(path_txt, path) for path in sharp_list_valid]

    trues = [os.path.isfile(file) for file in blur_list +
                sharp_list + blur_list_valid + sharp_list_valid]

    for true in trues:
        if true != True:
            print('Non valid route in RBlur!')

    # Print the number of images of the datasets
    print('Images in the subsets: \n')
    print("    -Images in the GOPRO_train folder: ", len(list_blur), len(list_sharp))
    print("    -Images in the GOPRO_test folder : ", len(list_blur_valid), len(list_sharp_valid))
    print("    -Images in the RBlur_train folder: ", len(blur_list), len(sharp_list))
    print("    -Images in the RBlur_test folder : ", len(blur_list_valid), len(sharp_list_valid))



    # Now we fuse the two datasets: For the evaluation we only use GOPRO
    list_blur  = list_blur + blur_list
    list_sharp = list_sharp + sharp_list
    
    list_blur_valid  = blur_list_valid
    list_sharp_valid = sharp_list_valid 

    print('Total number of images:')
    print('     - Train:', len(list_blur), len(list_sharp))
    print('     - Test: ', len(list_blur_valid), len(list_sharp_valid))

    # we random sort the lists using random_sort_pairs
    
    list_blur, list_sharp             = random_sort_pairs(list_blur, list_sharp)
    list_blur_valid, list_sharp_valid = random_sort_pairs(list_blur_valid, list_sharp_valid)

    image_blur = cv.imread(list_blur[0])
    image_sharp_valid = cv.imread(list_sharp[0])
    print(image_blur.shape, image_sharp_valid.shape)



    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
    tensor_transform = transforms.ToTensor()
    flip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
        transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
    ])

    # Load the datasets
    train_dataset = MyDataset_Crop(list_blur, list_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=cropsize,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    # # print(len(train_dataset))
    # # print(train_dataset[0])
    # # for high, low in train_dataset:
    # #     print(type(high))
    # #Load the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    # #test_loader = None

    return train_loader, test_loader

def main_dataset_realblur(
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512,
                       num_workers=1, crop_type='Random'):


    # def flatten_list_comprehension(matrix):
    #     return [item for row in matrix for item in row]

    #Now we read the images of RealBlur and append their paths to another lists

    path_txt = '/home/danfei/Python_Workspace/RealBlur'
    blur_list, sharp_list = [], []
    blur_list_valid, sharp_list_valid = [], []

    with open(os.path.join(path_txt, 'RealBlur_J_train_list.txt')) as file:
        for line in file:
            sharp, blur = line.strip().split()
            blur_list.append(blur)
            sharp_list.append(sharp)

    with open(os.path.join(path_txt, 'RealBlur_J_test_list.txt')) as file:
        for line in file:
            sharp, blur = line.strip().split()
            blur_list_valid.append(blur)
            sharp_list_valid.append(sharp)



    #print(len(blur_list), len(sharp_list), len(sharp_list_valid), len(blur_list_valid))

    blur_list  = [os.path.join(path_txt, path) for path in blur_list]
    sharp_list = [os.path.join(path_txt, path) for path in sharp_list]

    blur_list_valid  = [os.path.join(path_txt, path) for path in blur_list_valid]
    sharp_list_valid = [os.path.join(path_txt, path) for path in sharp_list_valid]

    trues = [os.path.isfile(file) for file in blur_list +
                sharp_list + blur_list_valid + sharp_list_valid]

    for true in trues:
        if true != True:
            print('Non valid route in RBlur!')

    # Print the number of images of the datasets
    print('Images in the subsets: \n')
    print("    -Images in the RBlur_train folder: ", len(blur_list), len(sharp_list))
    print("    -Images in the RBlur_test folder : ", len(blur_list_valid), len(sharp_list_valid))



    # Now we fuse the two datasets: For the evaluation we only use GOPRO
    list_blur  =  blur_list
    list_sharp = sharp_list
    
    list_blur_valid  = blur_list_valid
    list_sharp_valid = sharp_list_valid 

    print('Total number of images:')
    print('     - Train:', len(list_blur), len(list_sharp))
    print('     - Test: ', len(list_blur_valid), len(list_sharp_valid))

    # we random sort the lists using random_sort_pairs
    
    list_blur, list_sharp             = random_sort_pairs(list_blur, list_sharp)
    list_blur_valid, list_sharp_valid = random_sort_pairs(list_blur_valid, list_sharp_valid)

    image_blur = cv.imread(list_blur[0])
    image_sharp_valid = cv.imread(list_sharp[0])
    print(image_blur.shape, image_sharp_valid.shape)



    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
    tensor_transform = transforms.ToTensor()
    flip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
        transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
    ])

    # Load the datasets
    train_dataset = MyDataset_Crop(list_blur, list_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=cropsize,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    # # print(len(train_dataset))
    # # print(train_dataset[0])
    # # for high, low in train_dataset:
    # #     print(type(high))
    # #Load the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    # #test_loader = None

    return train_loader, test_loader

def main_dataset_synthetic(batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512):

    GLOBAL_PATH_TRAIN = '../bdd100k_nafnet/bdd100k_200_train_pseudoreal'

    blur = []
    sharp = []
    for path in os.listdir(GLOBAL_PATH_TRAIN):
        path_images = os.path.join(GLOBAL_PATH_TRAIN, path)
        path_blur = os.path.join(path_images, 'blur')
        path_sharp = os.path.join(path_images, 'gt')
        path_to_gt = os.path.join(path_sharp, os.listdir(path_sharp)[0])
        if os.path.isdir(path_blur):  # and os.path.isdir(path_sharp):
            imgs_blur = glob(os.path.join(path_blur, '*.jpg'))
            blur.append(imgs_blur)
            for img in imgs_blur:
                # we load the same photo for each blur in this folder
                sharp.append(path_to_gt)

    # now create the test dataset
    GLOBAL_PATH_TEST = '../bdd100k_nafnet/bdd100k_20_test_pseudoreal'

    blur_valid = []
    sharp_valid = []
    for path in os.listdir(GLOBAL_PATH_TEST):
        path_images = os.path.join(GLOBAL_PATH_TEST, path)
        path_blur = os.path.join(path_images, 'blur')
        path_sharp = os.path.join(path_images, 'gt')
        path_to_gt = os.path.join(path_sharp, os.listdir(path_sharp)[0])
        if os.path.isdir(path_blur):  # and os.path.isdir(path_sharp):
            imgs_blur = glob(os.path.join(path_blur, '*.jpg'))
            blur_valid.append(imgs_blur)
            for img in imgs_blur:
                # we load the same photo for each blur in this folder
                sharp_valid.append(path_to_gt)

    def flatten_list_comprehension(matrix):
        return [item for row in matrix for item in row]
    blur = flatten_list_comprehension(blur)
    blur_valid = flatten_list_comprehension(blur_valid)

    # check if all the image routes are correct
    trues = [os.path.isfile(file)
             for file in blur + sharp + blur_valid + sharp_valid]
    for true in trues:
        if true != True:
            print('Non valid route!')

    image_blur = cv.imread(blur[0])
    image_sharp_valid = cv.imread(sharp_valid[0])
    if verbose:
        print('Images shape and type: ')
        print(image_blur.shape, image_blur.dtype,
              image_sharp_valid.shape, image_sharp_valid.dtype)

        print('Images in the subsets: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(blur))
        print("    -Images in the PATH_LOW_VALID folder: ", len(blur_valid))

    # define the transforms applied to the image for training and testing (only tensor transform) when read
    # transforms from PIL to torchTensor, normalized to [0,1] and the correct permutations for torching working
    tensor_transform = transforms.ToTensor()
    flip_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
        transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
    ])

    # Load the datasets
    train_dataset = MyDataset_Crop(blur, sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type='Center')
    test_dataset = MyDataset_Crop(blur_valid, sharp_valid, cropsize=cropsize,
                                  tensor_transform=tensor_transform, test=True, crop_type='Center')

    # #Load the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                              num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                             num_workers=1, pin_memory=True, drop_last=False)
    # #test_loader = None

    return train_loader, test_loader


# ----------------------------
if __name__ == '__main__':

    train_test = main_dataset_realblur_gopro(train_path='../../../GOPRO_dataset/train', test_path='../../../GOPRO_dataset/test',
                       batch_size_train=4, batch_size_test=1, verbose=False, cropsize=512,
                       num_workers=1, crop_type='Center')
