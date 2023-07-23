# Created by j.sha on 27.01.2023

import os
from pathlib import Path

root = Path(__file__).parents[1]

# plot setting
txt_line_height = 0.06
txt_font_size = 8

pi_type = {'bk': 'bk_pred',
           'clu': 'clu_pred',
           'exp': 'exp_pred'}

# dim0: ness, dim1: suff, dim2: sn
score_type_index = {"ness": 0, "suff": 1, "sn": 2}
score_example_index = {"neg": 0, "pos": 1}

indices_x = [0]
indices_y = [1]
indices_z = [2]
indices_red = [3]
indices_green = [4]
indices_blue = [5]
indices_sphere = [6]
indices_cube = [7]
indices_prob = [8]

group_index = {
    "color": 0,
    "shape": 1,
    "position": 2
}

group_positions = ["x", "y", "z"]
group_color = ["red", "green", "blue"]
group_shapes = ["sphere", "cube"]
group_group_shapes = ["obj", "line", "circle", "conic"]
group_screen_positions = ["x_center_screen", "y_center_screen"]

# 0:2 center_x, center_z
# 2 slope
# 3 x_length
# 4 z_length
# 5 is_line
# 6 is_circle
# 7 probability
group_tensor_index = {
    'x': 0,
    'y': 1,
    'z': 2,
    'red': 3,
    'green': 4,
    'blue': 5,
    'sphere': 6,
    'cube': 7,
    'obj': 8,
    'line': 9,
    'circle': 10,
    'conic': 11,
    'x_length': 12,
    'y_length': 13,
    'z_length': 14,
    "x_center_screen": 15,
    "y_center_screen": 16,
    "screen_left_x": 17,
    "screen_left_y": 18,
    "screen_right_x": 19,
    "screen_right_y": 20,
    "axis_x": 21,
    "axis_z": 22,
    "screen_axis_x": 23,
    "screen_axis_z": 24,
    "color_counter": 25,
    "shape_counter": 26,
    "size": 27,
}

obj_positions = ["x", "y", "z"]
obj_color = ["red", "green", "blue"]
obj_shapes = ["sphere", "cube"]
obj_screen_positions = ["screen_x", "screen_y"]

obj_tensor_index = {
    'x': 0,
    'y': 1,
    'z': 2,
    'red': 3,
    'green': 4,
    'blue': 5,
    'sphere': 6,
    'cube': 7,
    'prob': 8,
    'screen_x': 9,
    'screen_y': 10
}

buffer_path = root / ".." / "storage"
data_path = root / "data"
if not os.path.exists(buffer_path):
    os.mkdir(buffer_path)

if __name__ == "__main__":
    print("root path: " + str(root))
