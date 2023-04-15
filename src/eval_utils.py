import config
import torch

from src import config

ness_index = config.score_type_index["ness"]
suff_index = config.score_type_index["suff"]
sn_index = config.score_type_index["sn"]


def is_sn(score):
    if score[sn_index] == 1:
        return True
    return False


def is_sn_th_good(score, threshold):
    if score[sn_index] > threshold:
        return True
    else:
        return False


def is_nc(score):
    if score[ness_index] == 1:
        return True
    else:
        return False


def is_nc_th_good(score, threshold):
    if score[ness_index] > threshold:
        return True
    else:
        return False


def is_sc(score):
    if score[suff_index] == 1:
        return True
    else:
        return False


def is_sc_th_good(score, threshold):
    if score[suff_index] > threshold:
        return True
    else:
        return False


def group_func(data):
    if len(data.shape) == 3:
        group_res = data.sum(dim=-2)
    else:
        raise ValueError
    return group_res


def eval_count_diff(data, dim):
    """ calculate the difference """

    mean = data.mean(dim=dim)
    diff_sum = torch.abs(data - mean).sum()
    possible_maximum = data.max() * data.shape[0]
    diff_sum_norm = 1 - (diff_sum / possible_maximum)
    return diff_sum_norm


def count_func(data, epsilon=1e-10):
    if len(data.shape) == 2:
        count_res = torch.sum(data / (data + epsilon), dim=-1)
    else:
        raise ValueError

    return count_res


def eval_group_diff(data, dim):
    mean = data.mean(dim=dim)
    diff_sum = torch.abs(data - mean).sum()
    possible_maximum = data.sum(dim=-1)[0] * data.shape[1] * data.shape[0]
    diff_sum_norm = diff_sum / possible_maximum
    return diff_sum_norm


def eval_data(data):
    sum_first = group_func(data)
    sum_second = count_func(sum_first)
    eval_first = eval_group_diff(sum_first, dim=0)
    eval_second = eval_count_diff(sum_second, dim=0)
    eval_res = torch.tensor([eval_first, eval_second])
    return eval_res


def eval_score(positive_score, negative_score):
    res_score = positive_score.pow(50) * (1 - negative_score.pow(50))

    return res_score


def cluster_objects(pattern_dict):
    position_clus = []
    # cluster by positions
    pattern_pos = pattern_dict["val_pos"]
    pattern_neg = pattern_dict["val_neg"]
    group_trace = torch.zeros(len(config.group_index.keys()))

    # color group
    eval_color_positive_res = eval_data(pattern_pos[:, :, config.indices_color])
    eval_color_negative_res = eval_data(pattern_neg[:, :, config.indices_color])
    eval_color_res = eval_score(eval_color_positive_res, eval_color_negative_res)

    # shape group
    eval_shape_positive_res = eval_data(pattern_pos[:, :, config.indices_shape])
    eval_shape_negative_res = eval_data(pattern_neg[:, :, config.indices_shape])
    eval_shape_res = eval_score(eval_shape_positive_res, eval_shape_negative_res)

    return position_clus
