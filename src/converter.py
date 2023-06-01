# Created by jing at 01.06.23
import torch
from sklearn.linear_model import LinearRegression

import config
from aitk.utils import eval_utils


def to_line_tensor(objs):
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

    line_tensor[tensor_index["color_counter"]] = eval_utils.op_count_nonzeros(colors.sum(dim=0), axis=0, epsilon=1e-10)
    line_tensor[tensor_index["shape_counter"]] = eval_utils.op_count_nonzeros(shapes.sum(dim=0), axis=0, epsilon=1e-10)

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
