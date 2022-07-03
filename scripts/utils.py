import torch.nn as nn
import torch
import torch.nn.functional as F


class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BinaryConv2d, self) \
                            .__init__(in_channels, out_channels, kernel_size)

    def forward(self, X):
        # Binarization of the weights
        self.weight.data = self.weight.fp.sign().add(0.5).sign()
        return F.conv2d(input=X,
                        weight=self.weight,
                        bias=False,
                        stride=1,
                        padding='same')


class MrbConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BinaryConv2d, self) \
                            .__init__(in_channels, out_channels, kernel_size)

    def forward(self, X):
        pass
