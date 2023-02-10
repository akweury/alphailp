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
        plt.plot(x, y, label=labels[i], lw=3)

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


def plot_scatter_chart(data_list, path, title=None, x_scale=None, y_scale=None,
                       sub_folder=None, labels=None,
                       x_label=None, y_label=None, show=False, log_y=False, log_x=False, cla_leg=False):
    no_of_colors = len(data_list)
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
             for j in range(no_of_colors)]

    plt.figure(figsize=(6, 6))
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
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.figure(figsize=(1000, 1000 * 0.618))
    if not os.path.exists(str(path)):
        os.mkdir(path)

    img_folder = path / f"{date_now}_{time_now}"
    output_folder = img_folder
    if not os.path.exists(str(img_folder)):
        os.mkdir(str(img_folder))

    if sub_folder is not None:
        if not os.path.exists(str(img_folder / sub_folder)):
            output_folder = img_folder / sub_folder
            os.mkdir(str(output_folder))
        else:
            output_folder = img_folder / sub_folder

    plt.savefig(str(output_folder / f"{title}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


def plot_scatter_heat_chart(data_list, path, title=None, x_scale=None, y_scale=None,
                            sub_folder=None, labels=None,
                            x_label=None, y_label=None, show=False, log_y=False, log_x=False, cla_leg=False):
    resolution = 2

    def heatmap2d(arr: np.ndarray):
        img = ax1.imshow(arr, cmap='binary')
        plt.colorbar(img)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    if title is not None:
        ax1.set_title(title)
    if y_label is not None:
        ax1.set_ylabel(y_label)
    if x_label is not None:
        ax1.set_xlabel(x_label)
    if log_y:
        ax1.set_yscale('log')
    if log_x:
        ax1.set_xscale('log')

    for i, data in enumerate(data_list):
        data_map = np.zeros(shape=[resolution, resolution])
        for index in range(len(data[0])):
            x_index = int(data[0][index] * resolution)
            y_index = int(data[1][index] * resolution)
            data_map[x_index, y_index] += 1

        heatmap2d(data_map)

    if labels is not None:
        fig.text(0.1, 0.05, labels,
                 bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)), fontsize=8)

    # plt.legend()

    # plt.figure(figsize=(1000, 1000 * 0.618))

    if not os.path.exists(str(path)):
        os.mkdir(path)

    img_folder = path / f"{date_now}_{time_now}"
    output_folder = img_folder
    if not os.path.exists(str(img_folder)):
        os.mkdir(str(img_folder))

    if sub_folder is not None:
        if not os.path.exists(str(img_folder / sub_folder)):
            output_folder = img_folder / sub_folder
            os.mkdir(str(output_folder))
        else:
            output_folder = img_folder / sub_folder

    plt.savefig(str(output_folder / f"{title}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()
