# Created by jing at 26.06.23

import os
from pathlib import Path

from data.featureMapExplainer.patterns import p_color

root = Path(__file__).parents[2]
buffer_path = root / ".." / "storage"
if not os.path.exists(buffer_path):
    os.mkdir(buffer_path)
data_path = buffer_path / "featureMapExplainer"
if not os.path.exists(data_path):
    os.mkdir(data_path)


def main():
    width = 64
    height = 64
    p_color.generate(data_path,width, height, train_num=100, test_num=20)
    print("done!")


if __name__ == "__main__":
    main()
