import torch
import torch.nn as nn
import numpy as numpy
import matplotlib.pyplot as plt

import dataset
import networks
import utils
from torchinfo import summary

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    hr_images, lr_images, mean_image = dataset.get_images(4)
    model = networks.VDSR_new(mean_image)
    model.apply(utils.init_weights)
    output_hr = model(lr_images)



if __name__ == "__main__":
    main()
