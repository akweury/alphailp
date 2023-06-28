# Created by jing at 27.06.23


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LogSoftmax

from data.featureMapExplainer.nets.layers import Conv


class Feep(nn.Module):
    def __init__(self, in_ch, label_size):
        super().__init__()
        self.__name__ = 'ncnn'
        out_ch = 2
        self.conv1 = Conv(in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = Conv(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2))
        # self.conv1 = nn.Conv2d(in_ch, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(out_ch * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, label_size)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        last_fm = self.conv2(x)
        x = torch.flatten(last_fm, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.logSoftmax(x)
        return x, last_fm
