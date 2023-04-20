# Created by shaji on 17-Apr-23
import math
import torch
import itertools
from sklearn.linear_model import LinearRegression

import config


def in_ranges(value, line_ranges):
    for min_v, max_v in line_ranges:
        if value < max_v and value > min_v:
            return True
    return False


def get_comb(data, comb_size):
    pattern_numbers = math.comb(data.shape[0], comb_size)
    indices = torch.zeros(size=(pattern_numbers, comb_size), dtype=torch.uint8)

    for ss_i, subset_indice in enumerate(itertools.combinations(data.tolist(), comb_size)):
        indices[ss_i] = torch.tensor(sorted(subset_indice), dtype=torch.uint8)
    return indices


def to_line_tensor(point_groups):
    # 0:2 center_x, center_z
    # 2 slope
    # 3 x_length
    # 4 z_length
    # 5 is_line
    # 6 is_circle
    # 7 probability
    tensor_index = config.group_tensor_index
    line_tensor = torch.zeros(len(tensor_index.keys()))
    line_tensor[tensor_index["x"]] = point_groups[:, 0].mean()
    line_tensor[tensor_index["y"]] = point_groups[:, 1].mean()
    line_tensor[tensor_index["z"]] = point_groups[:, 2].mean()
    line_tensor[tensor_index['red']] = 0
    line_tensor[tensor_index['green']] = 0
    line_tensor[tensor_index['blue']] = 0
    line_tensor[tensor_index['sphere']] = 0
    line_tensor[tensor_index['cube']] = 0

    line_model = LinearRegression().fit(point_groups[:, 0:1], point_groups[:, 2:])
    line_tensor[tensor_index["line"]] = 1 - torch.abs(
        torch.from_numpy(line_model.predict(point_groups[:, 0:1])) - point_groups[:, 2:]).sum() / point_groups.shape[0]

    line_tensor[tensor_index['circle']] = 0
    line_tensor[tensor_index["x_length"]] = point_groups[:, 0].max() - point_groups[:, 0].min()
    line_tensor[tensor_index["y_length"]] = point_groups[:, 1].max() - point_groups[:, 1].min()
    line_tensor[tensor_index["z_length"]] = point_groups[:, 2].max() - point_groups[:, 2].min()

    return line_tensor


def to_circle_tensor(point_groups, center, r):
    tensor_index = config.group_tensor_index
    circle_tensor = torch.zeros(len(tensor_index.keys()))
    circle_tensor[tensor_index["x"]] = center[0]
    circle_tensor[tensor_index["y"]] = point_groups[:, 1].mean()
    circle_tensor[tensor_index["z"]] = center[1]
    circle_tensor[tensor_index["red"]] = 0
    circle_tensor[tensor_index["green"]] = 0
    circle_tensor[tensor_index["blue"]] = 0
    circle_tensor[tensor_index["sphere"]] = 0
    circle_tensor[tensor_index["cube"]] = 0
    circle_tensor[tensor_index["line"]] = 0

    circle_tensor[tensor_index["circle"]] = 1 - torch.abs(
        torch.sqrt(((point_groups[:, [0, 2]] - center) ** 2).sum(dim=1)) - r).sum() / point_groups.shape[0]

    circle_tensor[tensor_index["x_length"]] = point_groups[:, 0].max() - point_groups[:, 0].min()
    circle_tensor[tensor_index["y_length"]] = point_groups[:, 1].max() - point_groups[:, 1].min()
    circle_tensor[tensor_index["z_length"]] = point_groups[:, 2].max() - point_groups[:, 2].min()



    return circle_tensor
