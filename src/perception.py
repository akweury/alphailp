# Created by shaji on 21-Mar-23
import numpy as np
import torch
import math
from aitk.utils import log_utils
import percept
from data_hide import vertex_normalization
import itertools
import config


def get_perception_predictions(args):
    # train_pos_loader, val_pos_loader, test_pos_loader = get_data_pos_loader(args)
    # train_neg_loader, val_neg_loader, test_neg_loader = get_data_neg_loader(args)
    # if args.dataset_type == "kandinsky":
    #     pm_val_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_val.pth.tar")
    #     pm_train_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_train.pth.tar")
    #     pm_test_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_test.pth.tar")
    #
    #     val_pos_pred, val_neg_pred = percept.eval_images(args, pm_val_res_file, args.device, val_pos_loader,
    #                                                      val_neg_loader)
    #     train_pos_pred, train_neg_pred = percept.eval_images(args, pm_train_res_file, args.device, train_pos_loader,
    #                                                          train_neg_loader)
    #     test_pos_pred, test_neg_pred = percept.eval_images(args, pm_test_res_file, args.device, test_pos_loader,
    #                                                        test_neg_loader)

    if args.dataset_type == "hide":
        train_pos_pred, train_neg_pred = get_pred_res(args, "train")
        test_pos_pred, test_neg_pred = get_pred_res(args, "test")
        val_pos_pred, val_neg_pred = get_pred_res(args, "val")

    else:
        raise ValueError

    log_utils.add_lines(f"==== positive image number: {len(val_pos_pred)}", args.log_file)
    log_utils.add_lines(f"==== negative image number: {len(val_neg_pred)}", args.log_file)
    pm_prediction_dict = {
        'val_pos': val_pos_pred,
        'val_neg': val_neg_pred,
        'train_pos': train_pos_pred,
        'train_neg': train_neg_pred,
        'test_pos': test_pos_pred,
        'test_neg': test_neg_pred
    }

    return pm_prediction_dict


def hough_transform(x, origin=None):
    # https://stats.stackexchange.com/questions/375787/how-to-cluster-parts-of-broken-line-made-of-points
    if origin is None:
        origin = [0, 0]

    x = np.transpose(x - origin)
    dx = np.vstack((np.apply_along_axis(np.diff, 0, x), [0.0, 0.0]))
    v = np.vstack((-dx[:, 1], dx[:, 0])).T
    n = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).reshape(-1, 1)
    res = np.column_stack((np.sum(x * (v / (n + 1e-20)), axis=1), np.arctan2(v[:, 1], v[:, 0]))).T
    return res


def extract_patterns(args, pm_prediction):
    positions = pm_prediction[:, :, :3]

    sub_pattern_numbers = 0
    for i in range(3, args.e + 1):
        sub_pattern_numbers += math.comb(args.e, i)

    sub_patterns = torch.zeros(size=(sub_pattern_numbers, positions.shape[0], positions.shape[1], positions.shape[2]))
    for i in range(3, args.e + 1):
        for ss_i, subset in enumerate(itertools.combinations(list(range(positions.shape[1])), i)):
            sub_patterns[ss_i, :, :len(subset), :] = positions[:, subset, :]

    return sub_patterns


def data_ordering(data):
    data_ordered = torch.zeros(data.shape)
    delta = data[:, :, :3].max(dim=1, keepdims=True)[0] - data[:, :, :3].min(dim=1, keepdims=True)[0]
    order_axis = torch.argmax(delta, dim=2)
    for data_i in range(len(data)):
        data_order_i = data[data_i,:,order_axis[data_i]].sort(dim=0)[1].squeeze(1)
        data_ordered[data_i] = data[data_i,data_order_i,:]

    return data_ordered


def get_pred_res(args, data_type):
    od_res = config.buffer_path / "hide" / f"{args.dataset}" / f"{args.dataset}_pm_res_{data_type}.pth.tar"
    pred_pos, pred_neg = percept.convert_data_to_tensor(args, od_res)

    # normalize the position
    pred_pos_norm = vertex_normalization(pred_pos)
    pred_neg_norm = vertex_normalization(pred_neg)

    # order the data by vertices (align the axis with higher delta)
    pred_pos_ordered = data_ordering(pred_pos_norm)
    pred_neg_ordered = data_ordering(pred_neg_norm)

    if args.top_data < len(pred_pos_ordered):
        pred_pos_ordered = pred_pos_ordered[:args.top_data]
        pred_neg_ordered = pred_neg_ordered[:args.top_data]

    return pred_pos_ordered, pred_neg_ordered
