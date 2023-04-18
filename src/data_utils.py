# Created by shaji on 17-Apr-23
import math
import torch
import itertools


def group_func(data):
    if len(data.shape) == 3:
        group_res = data.sum(dim=-2)
    else:
        raise ValueError
    return group_res


def count_func(data, epsilon=1e-10):
    if len(data.shape) == 2:
        count_res = torch.sum(data / (data + epsilon), dim=-1)
    else:
        raise ValueError

    return count_res


def eval_count_diff(data, dim):
    """ calculate the difference """

    mean = data.mean(dim=dim)
    diff_sum = torch.abs(data - mean).sum()
    possible_maximum = data.max() * data.shape[0]
    diff_sum_norm = 1 - (diff_sum / possible_maximum)
    return diff_sum_norm


def eval_group_diff(data, dim):
    mean = data.mean(dim=dim)
    diff_sum = torch.abs(data - mean).sum()
    possible_maximum = data.sum(dim=-1)[0] * data.shape[1] * data.shape[0]
    diff_sum_norm = diff_sum / possible_maximum
    return diff_sum_norm


def eval_score(positive_score, negative_score):
    res_score = positive_score.pow(50) * (1 - negative_score.pow(50))
    return res_score


def diff_error(data):
    return torch.abs(data - data.mean()).sum()


def in_ranges(value, line_ranges):
    for min_v, max_v in line_ranges:
        if value < max_v and value > min_v:
            return True
    return False


def even_distance(values):
    values_sort, _ = values.sort()

    values_sort_shift = torch.zeros(size=values_sort.shape)
    values_sort_shift[:-1] = values_sort[1:]
    values_sort_shift[-1] = values_sort[0]
    diff = (values_sort_shift - values_sort)[:-1]
    res = diff_error(diff)
    return res


def get_comb(data, comb_size):
    pattern_numbers = math.comb(data.shape[0], comb_size)
    indices = torch.zeros(size=(pattern_numbers, comb_size), dtype=torch.uint8)

    for ss_i, subset_indice in enumerate(itertools.combinations(data, comb_size)):
        indices[ss_i] = torch.tensor(subset_indice, dtype=torch.uint8)
    return indices


def calc_dist(points, centers):
    dist = torch.sqrt(torch.sum((points - centers) ** 2, dim=1, keepdim=True))
    return dist


def get_poly_area(points):
    # https://stackoverflow.com/a/30408825/8179152
    x = points[:, :, 0]
    y = points[:, :, 2]
    area = 0.5 * torch.abs((x * torch.roll(y, 1, 1)).sum(dim=1) - (y * torch.roll(x, 1, 1)).sum(dim=1))
    return area


def extend_point_groups(point_groups, extend_points):
    group_points_duplicate_all = point_groups.unsqueeze(0).repeat(extend_points.shape[0], 1, 1)
    group_points_candidate = torch.cat([group_points_duplicate_all, extend_points.unsqueeze(1)], dim=1)
    return group_points_candidate


def extend_line_group(group_index, points, error_th):
    point_groups = points[group_index]
    extra_index = torch.tensor(sorted(set(list(range(points.shape[0]))) - set(group_index)))
    extra_points = points[extra_index]
    point_groups_extended = extend_point_groups(point_groups, extra_points)
    is_line = predict_lines(point_groups_extended, error_th)

    passed_points = extra_points[is_line]
    passed_indices = extra_index[is_line]
    point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    point_groups_indices_new = torch.cat([torch.tensor(group_index), passed_indices])
    return point_groups_new, point_groups_indices_new


# def extend_line_group(group_index, objs, area_th):
#     point_groups = objs[group_index]
#     extra_index = torch.tensor(sorted(set(list(range(objs.shape[0]))) - set(group_index)))
#     extra_points = objs[extra_index]
#     point_groups_extended = extend_point_groups(point_groups, extra_points)
#     group_areas = get_poly_area(point_groups_extended)
#
#     passed_points = extra_points[group_areas < area_th]
#     passed_indices = extra_index[group_areas < area_th]
#     point_groups_new = torch.cat([point_groups, passed_points], dim=0)
#     point_groups_indices_new = torch.cat([torch.tensor(group_index), passed_indices])
#     group_areas_new = get_poly_area(point_groups_new.unsqueeze(0))
#     if group_areas_new < area_th:
#         return point_groups_new, point_groups_indices_new
#     else:
#         return None, None


def get_circle_error(c, r, points):
    dists = torch.sqrt(((points[:, [0, 2]] - c) ** 2).sum(1))
    return torch.abs(dists - r)


def get_circle(point_groups, collinear_th):
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


def extend_circle_group(group_index, points, error_th):
    point_groups = points[group_index]
    c, r = get_circle(point_groups, error_th)
    if c is None or r is None:
        return None, None
    extra_index = torch.tensor(sorted(set(list(range(points.shape[0]))) - set(group_index)))
    extra_points = points[extra_index]
    point_groups_extended = extend_point_groups(point_groups, extra_points)
    group_error = get_circle_error(c, r, extra_points)

    passed_points = extra_points[group_error < error_th]
    passed_indices = extra_index[group_error < error_th]
    point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    point_groups_indices_new = torch.cat([torch.tensor(group_index), passed_indices])
    return point_groups_new, point_groups_indices_new
