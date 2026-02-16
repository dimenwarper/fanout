import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Standard 2D convolution with square input and square kernel.
    """
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        return self.conv1(x)


batch_size = 256
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
