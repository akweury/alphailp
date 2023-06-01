# Created by shaji on 17-Apr-23
import math
import torch
import itertools

import config
from aitk.utils import eval_utils


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


def euclidean_distance(point_groups_screen, center):
    squared_distance = torch.sum(torch.square(point_groups_screen - torch.tensor(center)), dim=1)
    distance = torch.sqrt(squared_distance)
    return distance


def to_circle_tensor(point_groups, point_groups_screen, colors, shapes, center, r):
    tensor_index = config.group_tensor_index
    cir_tensor = torch.zeros(len(tensor_index.keys()))
    cir_tensor[tensor_index["x"]] = center[0]
    cir_tensor[tensor_index["y"]] = point_groups[:, 1].mean()
    cir_tensor[tensor_index["z"]] = center[1]

    cir_tensor[tensor_index["color_counter"]] = eval_utils.op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    cir_tensor[tensor_index["shape_counter"]] = eval_utils.op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

    cir_tensor[tensor_index["red"]] = 0
    cir_tensor[tensor_index["green"]] = 0
    cir_tensor[tensor_index["blue"]] = 0
    cir_tensor[tensor_index["sphere"]] = 0
    cir_tensor[tensor_index["cube"]] = 0
    cir_tensor[tensor_index["line"]] = 0

    cir_tensor[tensor_index["circle"]] = 1 - torch.abs(
        torch.sqrt(((point_groups[:, [0, 2]] - center) ** 2).sum(dim=1)) - r).sum() / point_groups.shape[0]

    cir_tensor[tensor_index["x_length"]] = point_groups[:, 0].max() - point_groups[:, 0].min()
    cir_tensor[tensor_index["y_length"]] = point_groups[:, 1].max() - point_groups[:, 1].min()
    cir_tensor[tensor_index["z_length"]] = point_groups[:, 2].max() - point_groups[:, 2].min()
    cir_tensor[tensor_index["x_center_screen"]] = point_groups_screen[:, 0].mean()
    cir_tensor[tensor_index["y_center_screen"]] = point_groups_screen[:, 1].mean()
    cir_tensor[tensor_index["screen_left_x"]] = 0
    cir_tensor[tensor_index["screen_left_y"]] = 0
    cir_tensor[tensor_index["screen_right_x"]] = 0
    cir_tensor[tensor_index["screen_right_y"]] = 0

    cir_tensor[tensor_index["radius"]] = euclidean_distance(point_groups[:, [0, 2]], (
        point_groups[:, 0].mean(), point_groups[:, 2].mean())).mean()

    cir_tensor[tensor_index["screen_radius"]] = euclidean_distance(point_groups_screen, (
        point_groups_screen[:, 0].mean(), point_groups_screen[:, 1].mean())).mean()

    # circle_tensor[tensor_index["x_length_screen"]] = point_groups_screen[:, 0].max() - point_groups_screen[:, 0].min()
    # circle_tensor[tensor_index["y_length_screen"]] = point_groups_screen[:, 1].max() - point_groups_screen[:, 1].min()

    return cir_tensor
