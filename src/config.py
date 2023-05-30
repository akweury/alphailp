# Created by j.sha on 27.01.2023

import os
from pathlib import Path



pi_type = {'bk': 'bk_pred',
          'clu': 'clu_pred',
          'exp': 'exp_pred'}

# dim0: ness, dim1: suff, dim2: sn
score_type_index = {"ness": 0, "suff": 1, "sn": 2}
score_example_index = {"neg": 0, "pos": 1}
root = Path(__file__).parents[1]

indices_position = [0, 1, 2]
indices_color = [3, 4, 5]
indices_shape = [6, 7]
indices_screen_position = [9, 10]

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
    'line': 8,
    'circle': 9,
    'x_length': 10,
    'y_length': 11,
    'z_length': 12,
    "x_center_screen": 13,
    "y_center_screen": 14,
    "screen_left_x": 15,
    "screen_left_y": 16,
    "screen_right_x": 17,
    "screen_right_y": 18,
    "radius": 19,
    "screen_radius": 20,
    "color_counter": 21,
    "shape_counter": 22
}






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

group_tenor_positions = [0, 1]
group_tensor_shapes = [5, 6]

buffer_path = root / ".." / "storage"
data_path = root / "data"
if not os.path.exists(buffer_path):
    os.mkdir(buffer_path)

if __name__ == "__main__":
    print("root path: " + str(root))
