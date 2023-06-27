# Created by jing at 26.06.23

import os
import datetime
import torch
import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def draw_line_chart(args, data_type, title=None, x_label=None, y_label=None, show=False, log_y=False, label=None,
                    cla_leg=False, loss_type="mse"):
    if data_type == "eval":
        data_1 = args.losses_eval
    else:
        data_1 = args.losses_train

    path = args.output_path
    epoch = args.epoch
    start_epoch = args.start_epoch

    x = np.arange(epoch - start_epoch)
    y = data_1[start_epoch:epoch].detach().numpy()
    x = x[y.nonzero()]
    y = y[y.nonzero()]
    plt.plot(x, y, label=label)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)
    if not os.path.exists(str(path)):
        os.mkdir(path)

    if loss_type == "mse":
        plt.savefig(str(Path(path) / f"{title}.png"))
    else:
        raise ValueError("loss type is not supported.")

    if show:
        plt.show()
    if cla_leg:
        plt.cla()
