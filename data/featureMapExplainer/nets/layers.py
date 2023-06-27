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

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.active_LeakyReLU = nn.LeakyReLU(0.01)
        self.active_ReLU = nn.ReLU()
        self.active_Sigmoid = nn.Sigmoid()
        self.active_Tanh = nn.Tanh()
        self.active_name = active_function
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.bn1(self.conv(x))

        if self.active_name == "LeakyReLU":
            return self.active_LeakyReLU(x)
        elif self.active_name == "Sigmoid":
            return self.active_Sigmoid(x)
        elif self.active_name == "ReLU":
            return self.active_ReLU(x)
        elif self.active_name == "Tanh":
            return self.active_Tanh(x)
        elif self.active_name == "":
            return x
        else:
            raise ValueError