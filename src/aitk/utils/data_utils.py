# Created by jing at 31.05.23
import itertools
import math
import torch
from sklearn.linear_model import LinearRegression

import config
from aitk.utils import eval_utils


def prop2index(props, g_type="group"):
    indices = []
    if g_type == "group":
        for prop in props:
            indices.append(config.group_tensor_index[prop])

    elif g_type == "object":
        for prop in props:
            indices.append(config.obj_tensor_index[prop])
    else:
        raise ValueError
    return indices


def get_comb(data, comb_size):
    pattern_numbers = math.comb(data.shape[0], comb_size)
    indices = torch.zeros(size=(pattern_numbers, comb_size), dtype=torch.uint8)

    for ss_i, subset_indice in enumerate(itertools.combinations(data.tolist(), comb_size)):
        indices[ss_i] = torch.tensor(sorted(subset_indice), dtype=torch.uint8)
    return indices


def in_ranges(value, line_ranges):
    for min_v, max_v in line_ranges:
        if value < max_v and value > min_v:
            return True
    return False


def euclidean_distance(point_groups_screen, center):
    squared_distance = torch.sum(torch.square(point_groups_screen - torch.tensor(center)), dim=1)
    distance = torch.sqrt(squared_distance)
    return distance


def to_line_tensor(objs, obj_indices, args, img_i):
    colors = objs[:, config.indices_color]
    shapes = objs[:, config.indices_shape]
    point_groups_screen = objs[:, config.indices_screen_position]

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

    line_tensor[tensor_index["color_counter"]] = op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    line_tensor[tensor_index["shape_counter"]] = op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

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

    if line_tensor[tensor_index["x_length"]] > line_tensor[tensor_index["z_length"]]:

        sorted_x, sorted_x_indices = point_groups_screen[:, 0].sort(dim=0)
        line_tensor[tensor_index["screen_left_x"]] = sorted_x[0]
        line_tensor[tensor_index["screen_left_y"]] = point_groups_screen[:, 1][sorted_x_indices[0]]
        line_tensor[tensor_index["screen_right_x"]] = sorted_x[-1]
        line_tensor[tensor_index["screen_right_y"]] = point_groups_screen[:, 1][sorted_x_indices[-1]]
    else:
        sorted_y, sorted_y_indices = point_groups_screen[:, 1].sort(dim=0)
        line_tensor[tensor_index["screen_left_y"]] = sorted_y[0]
        line_tensor[tensor_index["screen_left_x"]] = point_groups_screen[:, 0][sorted_y_indices[0]]
        line_tensor[tensor_index["screen_right_y"]] = sorted_y[-1]
        line_tensor[tensor_index["screen_right_x"]] = point_groups_screen[:, 0][sorted_y_indices[-1]]

    line_tensor[tensor_index["radius"]] = 0
    line_tensor[tensor_index["screen_radius"]] = 0
    line_tensor[tensor_index["size"]] = objs.shape[0]

    line_tensor = line_tensor.reshape(-1)
    line_used_objs = torch.zeros(args.n_obj, dtype=torch.bool)
    line_used_objs[obj_indices] = True

    return line_tensor, line_used_objs


def to_circle_tensor(args, objs, obj_indices, img_i):
    center, r = fit_circle(objs, args)

    colors = objs[:, config.indices_color]
    shapes = objs[:, config.indices_shape]
    point_groups_screen = objs[:, config.indices_screen_position]

    tensor_index = config.group_tensor_index
    cir_tensor = torch.zeros(len(tensor_index.keys()))
    cir_tensor[tensor_index["x"]] = center[0]
    cir_tensor[tensor_index["y"]] = objs[:, 1].mean()
    cir_tensor[tensor_index["z"]] = center[1]

    cir_tensor[tensor_index["color_counter"]] = op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    cir_tensor[tensor_index["shape_counter"]] = op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

    cir_tensor[tensor_index["red"]] = 0
    cir_tensor[tensor_index["green"]] = 0
    cir_tensor[tensor_index["blue"]] = 0
    cir_tensor[tensor_index["sphere"]] = 0
    cir_tensor[tensor_index["cube"]] = 0
    cir_tensor[tensor_index["line"]] = 0

    cir_tensor[tensor_index["circle"]] = 1 - torch.abs(
        torch.sqrt(((objs[:, [0, 2]] - center) ** 2).sum(dim=1)) - r).sum() / objs.shape[0]

    cir_tensor[tensor_index["x_length"]] = objs[:, 0].max() - objs[:, 0].min()
    cir_tensor[tensor_index["y_length"]] = objs[:, 1].max() - objs[:, 1].min()
    cir_tensor[tensor_index["z_length"]] = objs[:, 2].max() - objs[:, 2].min()
    cir_tensor[tensor_index["x_center_screen"]] = point_groups_screen[:, 0].mean()
    cir_tensor[tensor_index["y_center_screen"]] = point_groups_screen[:, 1].mean()
    cir_tensor[tensor_index["screen_left_x"]] = 0
    cir_tensor[tensor_index["screen_left_y"]] = 0
    cir_tensor[tensor_index["screen_right_x"]] = 0
    cir_tensor[tensor_index["screen_right_y"]] = 0

    cir_tensor[tensor_index["radius"]] = euclidean_distance(objs[:, [0, 2]], (
        objs[:, 0].mean(), objs[:, 2].mean())).mean()

    cir_tensor[tensor_index["screen_radius"]] = euclidean_distance(point_groups_screen, (
        point_groups_screen[:, 0].mean(), point_groups_screen[:, 1].mean())).mean()

    cir_tensor[tensor_index["size"]] = objs.shape[0]
    cir_tensor = cir_tensor.reshape(-1)
    # circle_tensor[tensor_index["x_length_screen"]] = point_groups_screen[:, 0].max() - point_groups_screen[:, 0].min()
    # circle_tensor[tensor_index["y_length_screen"]] = point_groups_screen[:, 1].max() - point_groups_screen[:, 1].min()
    # update point availabilities
    cir_used_objs = torch.zeros(args.n_obj, dtype=torch.bool)
    cir_used_objs[obj_indices] = True

    # print(f'(img {img_i}) circle group: {obj_indices}')

    return cir_tensor, cir_used_objs


def op_count_nonzeros(data, axis, epsilon):
    counter = (data / (data + epsilon)).sum(dim=axis)
    return counter


def fit_circle(data, args):
    min_group_indices = get_comb(torch.tensor(range(data.shape[0])), 3).tolist()
    centers = torch.zeros(len(min_group_indices), 2)
    radius = torch.zeros(len(min_group_indices))
    for g_i, group_indices in enumerate(min_group_indices):
        c, r = eval_utils.calc_circles(data[group_indices], args.cir_error_th)
        if c is not None:
            centers[g_i] = c
            radius[g_i] = r
    centers = centers.mean(dim=0)
    radius = radius.mean()
    return centers, radius
