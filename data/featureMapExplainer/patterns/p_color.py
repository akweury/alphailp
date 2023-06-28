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
    divider_x_position = 30

    # X
    x = np.zeros((w, h, 3), dtype=np.uint8)
    draw_color = np.random.randint(0, 256, 3, np.uint8)
    draw_percentage = 0.5
    draw_indices = np.random.choice(list(range(w)), int(draw_percentage * w))
    random_1 = np.random.choice(2)
    if random_1 == 0:
        x[:, :] = draw_color
    else:
        x[:, :] = draw_color

    # label y
    color_y = np.array([0, 0, 0])
    if draw_color.max() == draw_color[0]:
        color_y[0] = 1
        y = color_y
    elif draw_color.max() == draw_color[1]:
        color_y[1] = 1
        y = color_y
    elif draw_color.max() == draw_color[2]:
        color_y[2] = 1
        y = color_y
    else:
        raise ValueError

    if show:
        image_rgb = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        cv.imshow(f"p_color_{y}_{draw_color[0]}_{draw_color[1]}_{draw_color[2]}", image_rgb)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return x, y


def draw_rgb(w, h, save_path, show=False):
    divider_x_position = 30

    color_x = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_y = np.array([0, 0, 0])
    draw_percentage = 0.5
    for i in range(3):
        x = np.zeros((w, h, 3), dtype=np.uint8)
        draw_color = color_x[i]
        draw_indices = np.random.choice(list(range(w)), int(draw_percentage * w))

        random_1 = np.random.choice(2)
        if random_1 == 0:
            x[:, :] = draw_color
        else:
            x[:, :] = draw_color
        color_y[i] = 1
        y = color_y
        if show:
            image_rgb = cv.cvtColor(x, cv.COLOR_BGR2RGB)
            cv.imshow(f"p_color_{y}_{draw_color[0]}_{draw_color[1]}_{draw_color[2]}", image_rgb)
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
