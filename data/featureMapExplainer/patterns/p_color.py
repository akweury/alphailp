# Created by jing at 26.06.23
import os
from pathlib import Path
import numpy as np
import torch
import cv2 as cv

label_map = {
    0: "red",
    1: "green",
    2: "blue"
}


def draw(w, h, show=False):
    divider_x_position = w // 2
    left_color = np.random.randint(0, 256, 3, np.uint8)

    x = np.zeros((w, h, 3), dtype=np.uint8)
    x[:divider_x_position] = left_color
    color_y = np.array([0, 0, 0])
    if left_color.max() == left_color[0]:
        color_y[0] = 1
        y = color_y
    elif left_color.max() == left_color[1]:
        color_y[1] = 1
        y = color_y
    elif left_color.max() == left_color[2]:
        color_y[2] = 1
        y = color_y
    else:
        raise ValueError

    if show:
        image_rgb = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        cv.imshow(f"p_color_{y}_{left_color[0]}_{left_color[1]}_{left_color[2]}", image_rgb)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return x, y


def draw_rgb(w, h, save_path, show=False):
    divider_x_position = w // 2

    color_x = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_y = np.array([0, 0, 0])

    for i in range(3):
        x = np.zeros((w, h, 3), dtype=np.uint8)
        left_color = color_x[i]
        x[:divider_x_position] = left_color
        color_y[i] = 1
        y = color_y
        if show:
            image_rgb = cv.cvtColor(x, cv.COLOR_BGR2RGB)
            cv.imshow(f"p_color_{y}_{left_color[0]}_{left_color[1]}_{left_color[2]}", image_rgb)
            cv.waitKey(0)
            cv.destroyAllWindows()
        data = {"x": x, "y": y}
        torch.save(data, str(save_path / f"p_color_train_{i:05}.pth.tar"))


def generate(data_path, width, height, train_num, test_num):
    save_path = data_path / "p_color"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    draw_rgb(width, height, save_path)

    for i in range(train_num):
        x, y = draw(width, height)
        data = {"x": x, "y": y}
        torch.save(data, str(save_path / f"p_color_train_{i:05}.pth.tar"))
    for i in range(test_num):
        x, y = draw(width, height)
        data = {"x": x, "y": y}
        torch.save(data, str(save_path / f"p_color_test_{i:05}.pth.tar"))
