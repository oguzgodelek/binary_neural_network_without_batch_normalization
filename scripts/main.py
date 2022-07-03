import torch
import torch.nn as nn
import numpy as numpy
import matplotlib.pyplot as plt

import dataset


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    hr_images, lr_images = dataset.get_images(4)


if __name__ == "__main__":
    main()
