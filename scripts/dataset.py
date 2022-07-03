import os
import torch
import numpy as np
import cv2 as cv


def get_images(scale_factor):
    ''' Prepare and return image list
    Args:
        scale_factor(int): scale factor of interpolation
    Returns:
        image_array_hr: HR images
        image_array_lr: LR images
    '''
    # Read HR and LR training images from file
    hr_path_train = "../data/DIV2K/DIV2K_train_HR"
    lr_path_train = "../data/DIV2K/DIV2K_train_LR_bicubic/X"+str(scale_factor)

    dirlist_train_hr = os.listdir(hr_path_train)
    dirlist_train_hr.sort()
    # Get only first 100 images due to memory contraint
    dirlist_train_hr = dirlist_train_hr[:100]
    image_array_hr = [cv.imread(hr_path_train+"/"+image_path)
                      for image_path in dirlist_train_hr]

    dirlist_train_lr = os.listdir(lr_path_train)
    dirlist_train_lr.sort()
    dirlist_train_hr = dirlist_train_lr[:100]
    image_array_lr = [cv.imread(lr_path_train+"/"+image_path)
                      for image_path in dirlist_train_lr]

    for idx, image in enumerate(image_array_lr):
        image_array_lr[idx] = cv.resize(src=image,
                                        dsize=(image.shape[1]*scale_factor,
                                               image.shape[0]*scale_factor),
                                        interpolation=cv.INTER_CUBIC)
    return image_array_hr, image_array_lr


def preprocess_images():
    pass
