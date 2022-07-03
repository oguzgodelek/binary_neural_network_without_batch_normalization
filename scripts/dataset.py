import os
import torchvision.transforms as T
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def get_images(scale_factor, crop_window=48, crop_number=5):
    ''' Prepare and return image list
    Args:
        scale_factor(int): scale factor of interpolation
        crop_window(int): window size of random crops
        crop_number(int): how many cropped samples will be generated from
                          each image
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
    image_array_hr = [torch.from_numpy(img).permute(2, 0, 1)
                      for img in image_array_hr]
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
    image_array_lr = [torch.from_numpy(img).permute(2, 0, 1)
                      for img in image_array_lr]

    image_array_hr, image_array_lr = random_crops(image_array_hr,
                                                  image_array_lr,
                                                  crop_window, crop_number)
    plot_crops(image_array_hr[10:17])
    plot_crops(image_array_lr[10:17])
    return image_array_hr, image_array_lr


def random_crops(hr_imgs, lr_imgs, crop_size, crop_number):
    ''' Randomly select the specified number of square crops from each image
        with the crop size
    Args:
        hr_imgs(list[Tensor]): List of high resolution images
        lr_imgs(list[Tensor]): List of low resolution and interpolated images
        crop_size(int): the function generates (crop_size x crop_size) random
                        crops
        crop_number(int): how many cropped samples will be generated from
                          each image
    Returns:
        image_array_hr: cropped HR samples
        image_array_lr: cropped LR samples
    '''
    hr_samples = []
    lr_samples = []
    while len(hr_imgs) != 0:
        hr = hr_imgs[0]
        lr = lr_imgs[0]
        for _ in range(crop_number):
            random_h = np.random.randint(0, hr.size(dim=1)-crop_size-1)
            random_w = np.random.randint(0, hr.size(dim=2)-crop_size-1)
            hr_samples.append(hr[:, random_h:random_h+crop_size,
                                 random_w:random_w+crop_size])
            lr_samples.append(lr[:, random_h:random_h+crop_size,
                                 random_w:random_w+crop_size])
        hr_imgs.pop(0)
        lr_imgs.pop(0)

    return hr_samples, lr_samples


def plot_crops(imgs):
    '''  This function is used for plotting 5 crops to test the crops
    Args:
        imgs(list[Tensor]): Image list to be plotted

    Returns:
        None
    '''
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row =  row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img.permute(1, 2, 0)))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()
