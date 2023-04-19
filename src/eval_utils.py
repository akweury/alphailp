import torch

import config


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


def check_clu_result(clu_result):
    is_done = False
    for pred, res in clu_result.items():
        if res["result"] > 0.99:
            is_done = True
            break
    return is_done


def get_circle_error(c, r, points):
    dists = torch.sqrt(((points[:, [0, 2]] - c) ** 2).sum(1))
    return torch.abs(dists - r)


def eval_score(positive_score, negative_score):
    res_score = positive_score.pow(50) * (1 - negative_score.pow(50))
    return res_score




def eval_group_diff(data, dim):
    mean = data.mean(dim=dim)
    diff_sum = torch.abs(data - mean).sum()
    possible_maximum = data.sum(dim=-1)[0] * data.shape[1] * data.shape[0]
    diff_sum_norm = diff_sum / possible_maximum
    return diff_sum_norm


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


def group_func(data):
    if len(data.shape) == 3:
        group_res = data.sum(dim=-2)
    else:
        raise ValueError
    return group_res


def predict_circles(point_groups, collinear_th):
    # https://math.stackexchange.com/a/3503338
    complex_point_real = point_groups[:, 0]
    complex_point_imag = point_groups[:, 2]

    complex_points = torch.complex(complex_point_real, complex_point_imag)

    a, b, c = complex_points[0], complex_points[1], complex_points[2]
    if torch.abs(a - b).sum() < collinear_th or torch.abs(b - c).sum() < collinear_th or torch.abs(
            a - c).sum() < collinear_th:
        return None, None

    def f(z):
        return (z - a) / (b - a)

    def f_inv(w):
        return a + (b - a) * w

    w3 = f(c)
    if torch.abs(w3.imag) < collinear_th:
        # print("collinear point groups")
        return None, None
    center_complex = f_inv((w3 - w3 * w3.conj()) / (w3 - w3.conj()))
    r = torch.abs(a - center_complex)
    center = torch.tensor([center_complex.real, center_complex.imag])
    return center, r


def predict_lines(point_groups, collinear_th):
    # https://math.stackexchange.com/a/3503338
    complex_point_real = point_groups[:, :, 0]
    complex_point_imag = point_groups[:, :, 2]

    complex_points = torch.complex(complex_point_real, complex_point_imag)

    a, b, c = complex_points[:, 0], complex_points[:, 1], complex_points[:, 2]
    if torch.abs(a - b).sum() < collinear_th or torch.abs(b - c).sum() < collinear_th or torch.abs(
            a - c).sum() < collinear_th:
        # print("two points overlap with each other.")
        return None

    def f(z):
        return (z - a) / (b - a)

    w3 = f(c)
    # print("collinear point groups")
    collinearities = torch.abs(w3.imag)
    is_collinear = collinearities < collinear_th
    return is_collinear
