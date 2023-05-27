# Created by shaji on 17-Apr-23
import math
import torch
import itertools
from sklearn.linear_model import LinearRegression

import config
import eval_utils


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


def to_line_tensor(objs):

    colors = objs[:, config.indices_color]
    shapes = objs[:, config.indices_shape]
    point_groups_screen = objs[:,config.indices_screen_position]

    colors_normalized = colors.sum(dim=0) / colors.shape[0]
    shapes_normalized = shapes.sum(dim=0) / shapes.shape[0]
    # 0:2 center_x, center_z
    # 2 slope
    # 3 x_length
    # 4 z_length
    # 5 is_line
    # 6 is_circle
    # 7 probability
    tensor_index = config.group_tensor_index
    line_tensor = torch.zeros(len(tensor_index.keys()))
    line_tensor[tensor_index["x"]] = objs[:, 0].mean()
    line_tensor[tensor_index["y"]] = objs[:, 1].mean()
    line_tensor[tensor_index["z"]] = objs[:, 2].mean()

    line_tensor[tensor_index["color_counter"]] = eval_utils.count_func(colors.sum(dim=0).unsqueeze(0))
    line_tensor[tensor_index["shape_counter"]] = eval_utils.count_func(shapes.sum(dim=0).unsqueeze(0))

    colors_normalized[colors_normalized < 0.99] = 0
    line_tensor[tensor_index['red']] = colors_normalized[0]
    line_tensor[tensor_index['green']] = colors_normalized[1]
    line_tensor[tensor_index['blue']] = colors_normalized[2]

    shapes_normalized[shapes_normalized < 0.99] = 0
    line_tensor[tensor_index['sphere']] = shapes_normalized[0]
    line_tensor[tensor_index['cube']] = shapes_normalized[1]

    line_model = LinearRegression().fit(objs[:, 0:1], objs[:, 2:3])
    line_tensor[tensor_index["line"]] = 1 - torch.abs(
        torch.from_numpy(line_model.predict(objs[:, 0:1])) - objs[:, 2:3]).sum() / objs.shape[0]

    line_tensor[tensor_index['circle']] = 0
    line_tensor[tensor_index["x_length"]] = objs[:, 0].max() - objs[:, 0].min()
    line_tensor[tensor_index["y_length"]] = objs[:, 1].max() - objs[:, 1].min()
    line_tensor[tensor_index["z_length"]] = objs[:, 2].max() - objs[:, 2].min()

    line_tensor[tensor_index["x_center_screen"]] = point_groups_screen[:, 0].mean()
    line_tensor[tensor_index["y_center_screen"]] = point_groups_screen[:, 1].mean()
    sorted_x, sorted_x_indices = point_groups_screen[:, 0].sort(dim=0)

    line_tensor[tensor_index["screen_left_x"]] = sorted_x[0]
    line_tensor[tensor_index["screen_left_y"]] = point_groups_screen[:, 1][sorted_x_indices[0]]
    line_tensor[tensor_index["screen_right_x"]] = sorted_x[-1]
    line_tensor[tensor_index["screen_right_y"]] = point_groups_screen[:, 1][sorted_x_indices[-1]]

    line_tensor[tensor_index["radius"]] = 0
    line_tensor[tensor_index["screen_radius"]] = 0

    return line_tensor


def euclidean_distance(point_groups_screen, center):
    squared_distance = torch.sum(torch.square(point_groups_screen - torch.tensor(center)), dim=1)
    distance = torch.sqrt(squared_distance)
    return distance


def to_circle_tensor(point_groups, point_groups_screen, colors, shapes, center, r):
    tensor_index = config.group_tensor_index
    circle_tensor = torch.zeros(len(tensor_index.keys()))
    circle_tensor[tensor_index["x"]] = center[0]
    circle_tensor[tensor_index["y"]] = point_groups[:, 1].mean()
    circle_tensor[tensor_index["z"]] = center[1]

    circle_tensor[tensor_index["color_counter"]] = eval_utils.count_func(colors.sum(dim=0).unsqueeze(0))
    circle_tensor[tensor_index["shape_counter"]] = eval_utils.count_func(shapes.sum(dim=0).unsqueeze(0))

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
    circle_tensor[tensor_index["x_center_screen"]] = point_groups_screen[:, 0].mean()
    circle_tensor[tensor_index["y_center_screen"]] = point_groups_screen[:, 1].mean()
    circle_tensor[tensor_index["screen_left_x"]] = 0
    circle_tensor[tensor_index["screen_left_y"]] = 0
    circle_tensor[tensor_index["screen_right_x"]] = 0
    circle_tensor[tensor_index["screen_right_y"]] = 0

    circle_tensor[tensor_index["radius"]] = euclidean_distance(point_groups[:, [0, 2]], (
        point_groups[:, 0].mean(), point_groups[:, 2].mean())).mean()

    circle_tensor[tensor_index["screen_radius"]] = euclidean_distance(point_groups_screen, (
        point_groups_screen[:, 0].mean(), point_groups_screen[:, 1].mean())).mean()

    # circle_tensor[tensor_index["x_length_screen"]] = point_groups_screen[:, 0].max() - point_groups_screen[:, 0].min()
    # circle_tensor[tensor_index["y_length_screen"]] = point_groups_screen[:, 1].max() - point_groups_screen[:, 1].min()

    return circle_tensor
