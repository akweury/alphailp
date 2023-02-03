# Create by J.Sha on 02.02.2023
import os
import datetime
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def plot_line_chart(data, path, labels, x=None, title=None, x_scale=None, y_scale=None, y_label=None, show=False,
                    log_y=False, cla_leg=False):
    if data.shape[1] <= 1:
        return

    if y_scale is None:
        y_scale = [1, 1]
    if x_scale is None:
        x_scale = [1, 1]

    for i, row in enumerate(data):
        if x is None:
            x = np.arange(row.shape[0]) * x_scale[1]
        y = row
        plt.plot(x, y, label=labels[i], lw=5)

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)

    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{y_label}_{date_now}_{time_now}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()

    if show:
        plt.show()


def plot_scatter_chart(data_list, path, title=None, x_scale=None, y_scale=None, labels=None,
                       x_label=None, y_label=None, show=False, log_y=False, log_x=False, cla_leg=False):
    no_of_colors = len(data_list)
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
             for j in range(no_of_colors)]

    plt.figure(figsize=(10, 6))
    for i, data in enumerate(data_list):
        data_x = data[0]
        data_y = data[1]
        plt.scatter(data_x, data_y, label=i, c=color[i])

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

    if log_y:
        plt.yscale('log')
    if log_x:
        plt.xscale('log')

    plt.legend()

    # plt.figure(figsize=(1000, 1000 * 0.618))
    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{date_now}_{time_now}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


def plot_scatter_heat_chart(data_list, path, title=None, x_scale=None, y_scale=None, labels=None,
                            x_label=None, y_label=None, show=False, log_y=False, log_x=False, cla_leg=False):
    def heatmap2d(arr: np.ndarray):
        plt.imshow(arr, cmap='binary')
        plt.colorbar()

    resolution = 2
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(data_list):
        data_map = np.zeros(shape=[resolution, resolution])
        for index in range(len(data[0])):
            x_index = int(data[0][index] * resolution)
            y_index = int(data[1][index] * resolution)
            data_map[x_index, y_index] += 1

        heatmap2d(data_map)

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

    if log_y:
        plt.yscale('log')
    if log_x:
        plt.xscale('log')

    if labels is not None:
        plt.text(-0.4, 1.4, labels,
                 bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)),
                 fontsize=8,
                 )

    # plt.legend()

    # plt.figure(figsize=(1000, 1000 * 0.618))
    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{date_now}_{time_now}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()
