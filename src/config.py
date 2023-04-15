# Created by j.sha on 27.01.2023

import os
from pathlib import Path

# dim0: ness, dim1: suff, dim2: sn
score_type_index = {"ness": 0, "suff": 1, "sn": 2}
score_example_index = {"neg": 0, "pos": 1}
root = Path(__file__).parents[1]

indices_position = [0, 1, 2]
indices_color = [3, 4, 5]
indices_shape = [6, 7]

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

buffer_path = root / ".." / "buffer"
data_path = root / "data"
if not os.path.exists(buffer_path):
    os.mkdir(buffer_path)
if __name__ == "__main__":
    print("root path: " + str(root))
