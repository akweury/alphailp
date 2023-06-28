# Created by jing at 26.06.23

import os
import datetime

import torch
import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


# https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
def concat_vh(list_2d):
    """
    show image in a 2d array
    :param list_2d: 2d array with image element
    :return: concatenated 2d image array
    """
    # return final image
    return cv.vconcat([cv.hconcat(list_h)
                       for list_h in list_2d])


def normalize(numpy_array):
    if numpy_array.ndim != 2:
        if numpy_array.shape == (512, 512):
            numpy_array = numpy_array.reshape(512, 512, 1)
    else:
        min, max = numpy_array.min(), numpy_array.max()
        if min == max:
            return numpy_array, 0, 1

    numpy_array = (numpy_array - min).astype(np.float32) /255
    return numpy_array, min, max


def normalize3channel(numpy_array):
    mins, maxs = [], []
    if numpy_array.ndim != 3:
        raise ValueError
    h, w, c = numpy_array.shape
    for i in range(c):
        numpy_array[:, :, i], min, max = normalize(numpy_array[:, :, i])
        mins.append(min)
        maxs.append(max)
    return numpy_array, mins, maxs


def convert_to_8bit(img_scaled):
    img_32bit = cv.normalize(img_scaled, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_32bit


# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def draw_line_chart(args, data_type, title=None, x_label=None, y_label=None, show=False, log_y=False, label=None,
                    cla_leg=False, loss_type="mse"):
    if data_type == "eval":
        data_1 = args.losses_eval
    elif data_type == "train":
        data_1 = args.losses_train
    elif data_type == "accuracy":
        data_1 = args.accuracy
    else:
        raise ValueError
    path = args.analysis_path
    epoch = args.epoch
    start_epoch = args.start_epoch

    x = np.arange(epoch - start_epoch)
    y = data_1[start_epoch:epoch].detach().numpy()
    x = x[y.nonzero()]
    y = y[y.nonzero()]
    plt.plot(x, y, label=label)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)
    if not os.path.exists(str(path)):
        os.mkdir(path)

    if loss_type == "mse":
        plt.savefig(str(Path(path) / f"{title}.png"))
    else:
        raise ValueError("loss type is not supported.")

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


def image_frame(fm_8bit, f_width):
    fm_8bit[:, :f_width] = 255
    fm_8bit[:, fm_8bit.shape[0] - f_width:] = 255
    fm_8bit[:f_width, :] = 255
    fm_8bit[fm_8bit.shape[0] - f_width:, :] = 255
    return fm_8bit


def image_divider(fm_8bit, f_width):
    fm_8bit[int(fm_8bit.shape[0] / 2 - f_width):int(fm_8bit.shape[0] / 2 + f_width), :] = 255
    return fm_8bit


def addText(img, text, pos='upper_left', font_size=1.6, color=(255, 255, 255), thickness=1):
    h, w = img.shape[:2]
    if pos == 'upper_left':
        position = (10, 50)
    elif pos == 'upper_right':
        position = (w - 250, 80)
    elif pos == 'lower_right':
        position = (h - 200, w - 20)
    elif pos == 'lower_left':
        position = (10, w - 20)
    else:
        raise ValueError('unsupported position to put text in the image.')

    cv.putText(img, text=text, org=position,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=thickness, lineType=cv.LINE_AA)
