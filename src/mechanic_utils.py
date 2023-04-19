import torch

import config
from data_utils import to_line_tensor, get_comb, to_circle_tensor
from eval_utils import eval_group_diff, eval_count_diff, count_func, group_func, get_circle_error, eval_score
from src.eval_utils import predict_circles, predict_lines


def detect_line_groups(args, data):
    line_tensors = torch.zeros(data.shape[0], args.group_e, len(config.group_tensor_index.keys()))
    for data_i in range(data.shape[0]):
        exist_combs = []
        group_indices = get_comb(torch.tensor(range(data[data_i].shape[0])), 2).tolist()
        tensor_counter = 0
        line_tensor_candidates = torch.zeros(args.group_e, len(config.group_tensor_index.keys()))
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                point_groups, point_indices = extend_line_group(group_index, data[data_i], args.error_th)
                if point_groups is not None and point_groups.shape[0] >= args.line_group_min_sz:
                    line_tensor = to_line_tensor(point_groups).reshape(1, -1)
                    line_tensor_candidates = torch.cat([line_tensor_candidates, line_tensor], dim=0)
                    exist_combs += get_comb(point_indices, 2).tolist()
                    tensor_counter += 1
        _, prob_indices = line_tensor_candidates[:, config.group_tensor_index["probability"]].sort(descending=True)
        prob_indices = prob_indices[:args.group_e]
        line_tensors[data_i] = line_tensor_candidates[prob_indices]
    return line_tensors


def detect_circle_groups(args, data):
    circle_tensors = torch.zeros(data.shape[0], args.group_e, len(config.group_tensor_index.keys()))
    for data_i in range(data.shape[0]):
        exist_combs = []
        tensor_counter = 0
        group_indices = get_comb(torch.tensor(range(data[data_i].shape[0])), 3).tolist()
        circle_tensor_candidates = torch.zeros(args.group_e, len(config.group_tensor_index.keys()))
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                p_groups, p_indices, center, r = extend_circle_group(group_index, data[data_i], args.error_th)
                if p_groups is not None and p_groups.shape[0] >= args.cir_group_min_sz:
                    circle_tensor = to_circle_tensor(p_groups, center=center, r=r).reshape(1, -1)
                    circle_tensor_candidates = torch.cat([circle_tensor_candidates, circle_tensor], dim=0)
                    exist_combs += get_comb(p_indices, 3).tolist()
                    tensor_counter += 1
        _, prob_indices = circle_tensor_candidates[:, config.group_tensor_index["probability"]].sort(descending=True)
        prob_indices = prob_indices[:args.group_e]
        circle_tensors[data_i] = circle_tensor_candidates[prob_indices]
    return circle_tensors


def extend_line_group(group_index, points, error_th):
    point_groups = points[group_index]
    extra_index = torch.tensor(sorted(set(list(range(points.shape[0]))) - set(group_index)))
    extra_points = points[extra_index]
    point_groups_extended = extend_groups(point_groups, extra_points)
    is_line = predict_lines(point_groups_extended, error_th)

    passed_points = extra_points[is_line]
    passed_indices = extra_index[is_line]
    point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    point_groups_indices_new = torch.cat([torch.tensor(group_index), passed_indices])
    return point_groups_new, point_groups_indices_new


def extend_circle_group(group_index, points, error_th):
    point_groups = points[group_index]
    c, r = predict_circles(point_groups, error_th)
    if c is None or r is None:
        return None, None, None, None
    extra_index = torch.tensor(sorted(set(list(range(points.shape[0]))) - set(group_index)))
    extra_points = points[extra_index]
    point_groups_extended = extend_groups(point_groups, extra_points)
    group_error = get_circle_error(c, r, extra_points)
    passed_points = extra_points[group_error < error_th]
    passed_indices = extra_index[group_error < error_th]
    point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    point_groups_indices_new = torch.cat([torch.tensor(group_index), passed_indices])
    return point_groups_new, point_groups_indices_new, c, r


def extend_groups(point_groups, extend_points):
    group_points_duplicate_all = point_groups.unsqueeze(0).repeat(extend_points.shape[0], 1, 1)
    group_points_candidate = torch.cat([group_points_duplicate_all, extend_points.unsqueeze(1)], dim=1)
    return group_points_candidate


def eval_single_group(data_pos, data_neg):
    group_pos_res, score_lines_1, score_lines_2 = eval_data(data_pos)
    group_neg_res, _, _ = eval_data(data_neg)
    res = eval_score(group_pos_res, group_neg_res)
    return res


def eval_data(data):
    sum_first = group_func(data)
    eval_first = eval_group_diff(sum_first, dim=0)
    sum_second = count_func(sum_first)
    eval_second = eval_count_diff(sum_second, dim=0)
    eval_res = torch.tensor([eval_first, eval_second])
    return eval_res, sum_first[0], sum_second[0]


def test_groups_on_one_image(data, val_result):
    test_score = 1
    eval_color_positive_res, score_color_1, score_color_2 = eval_data(data[:, :, config.indices_color])
    eval_shape_positive_res, score_shape_1, score_shape_2 = eval_data(data[:, :, config.indices_shape])

    return test_score
