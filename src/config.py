# Created by j.sha on 27.01.2023

import os
from pathlib import Path

root = Path(__file__).parents[1]

trivial_preds_dict = [
    ["at_area_0", "at_area_1", "at_area_2", "at_area_3", "at_area_4", "at_area_5", "at_area_6", "at_area_7"],
    ["diff_shape_pair", "same_shape_pair"],
    ["diff_color_pair", "same_color_pair"]
]

buffer_path = root / "src" / "runs" / "buffer"
if __name__ == "__main__":
    print("root path: " + str(root))
