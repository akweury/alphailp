# Created by jing at 27.06.23
import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from data.featureMapExplainer.nets import feep


class CNN(nn.Module):
    def __init__(self, input_ch, label_size):
        super().__init__()
        self.__name__ = 'p_color'

        # input confidence estimation network
        self.feep = feep.Feep(input_ch, label_size)

    def forward(self, x0):
        xout = self.feep(x0)
        return xout
