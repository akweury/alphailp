# Created by jing at 27.06.23
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


class Conv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True, active_function="LeakyReLU"):
        # Call _ConvNd constructor
        super(Conv, self).__init__(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, False, (0, 0),
                                   groups, bias, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        return x
