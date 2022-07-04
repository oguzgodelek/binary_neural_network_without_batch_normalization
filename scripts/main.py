import torch
import torch.nn as nn
import numpy as numpy
import matplotlib.pyplot as plt

import dataset
import networks


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    hr_images, lr_images, mean_image = dataset.get_images(4)
    model = networks.VDSR(mean_image)
    model.apply(networks.init_weights)
    output_hr = model(lr_images)


if __name__ == "__main__":
    main()
