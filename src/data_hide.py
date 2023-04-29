import glob
import os

import torch

import config


def vertex_normalization(data):
    if len(data.shape) != 3:
        raise ValueError

    ax = 0
    data[:, :, :3] = (data[:, :, :3] - data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[0]) / (
            data[:, :, ax:ax + 1].max(axis=1, keepdims=True)[0] - data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[
        0] + 1e-10)

    ax = 2
    data[:, :, ax] = data[:, :, ax] - data[:, :, ax].min(axis=1, keepdims=True)[0]
    # for i in range(len(data)):
    #     data_plot = np.zeros(shape=(5, 2))
    #     data_plot[:, 0] = data[i, :5, 0]
    #     data_plot[:, 1] = data[i, :5, 2]
    #     chart_utils.plot_scatter_chart(data_plot, config.buffer_path / "hide", show=True, title=f"{i}")
    return data


def get_image_names(args):
    image_root = config.buffer_path / args.dataset_type / args.dataset
    image_name_dict = {}
    for data_mode in ['test', 'train', 'val']:
        tar_file = image_root / f"{args.dataset}_pm_res_{data_mode}.pth.tar"
        if not os.path.exists(tar_file):
            raise FileNotFoundError
        tensor_dict = torch.load(tar_file)

        image_name_dict[data_mode] = {}
        image_name_dict[data_mode]["true"] = tensor_dict["pos_names"]
        image_name_dict[data_mode]["false"] = tensor_dict["neg_names"]
        if len(image_name_dict[data_mode]["true"]) == 0 or len(image_name_dict[data_mode]["false"]) == 0:
            raise ValueError
    args.image_name_dict = image_name_dict
