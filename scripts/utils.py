import torch
import torch.nn as nn


def init_weights(module):
    if(isinstance(module, nn.Conv2d)):
        nn.init.xavier_uniform(module.weight)


def train_with_BAM(model, lr_images, hr_images):
    pass
