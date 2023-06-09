# Created by jing at 09.06.23
import copy
import os

import torch

import aitk.utils.data_utils
import config

from aitk.utils import eval_utils
from aitk.utils.data_utils import to_circle_tensor, get_comb
from aitk.utils.visual_utils import visual_lines


def detect_line_groups(args, obj_tensors, data_type):
    """
    input: object tensors
    output: group tensors
    """

    line_indices, line_groups = detect_lines(args, obj_tensors)
    line_tensors, line_tensors_indices = encode_lines(args, obj_tensors, line_indices, line_groups)

    # logic: prune strategy
    pruned_tensors, pruned_tensors_indices = prune_lines(args, line_tensors, line_tensors_indices)

    visual_lines(args, pruned_tensors, pruned_tensors_indices, data_type)

    return pruned_tensors, pruned_tensors_indices


def detect_circle_groups(args, percept_dict):
    point_data = percept_dict[:, :, config.indices_position]
    color_data = percept_dict[:, :, config.indices_color]
    shape_data = percept_dict[:, :, config.indices_shape]
    point_screen_data = percept_dict[:, :, config.indices_screen_position]

    used_objs = torch.zeros(point_data.shape[0], args.group_e, point_data.shape[1], dtype=torch.bool)
    circle_tensors = torch.zeros(point_data.shape[0], args.e, len(config.group_tensor_index.keys()))
    for data_i in range(point_data.shape[0]):
        exist_combs = []
        tensor_counter = 0
        group_indices = get_comb(torch.tensor(range(point_data[data_i].shape[0])), 3).tolist()
        circle_tensor_candidates = torch.zeros(args.e, len(config.group_tensor_index.keys()))
        groups_used_objs = torch.zeros(args.group_e, point_data.shape[1], dtype=torch.bool)

        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                p_groups, p_indices, center, r = extend_circle_group(group_index, point_data[data_i], args)
                if p_groups is not None and p_groups.shape[0] >= args.cir_group_min_sz:
                    colors = color_data[data_i][p_indices]
                    shapes = shape_data[data_i][p_indices]
                    p_groups_screen = point_screen_data[data_i][p_indices]
                    circle_tensor = to_circle_tensor(p_groups, p_groups_screen, colors, shapes, center=center,
                                                     r=r).reshape(1, -1)
                    circle_tensor_candidates = torch.cat([circle_tensor_candidates, circle_tensor], dim=0)

                    cir_used_objs = torch.zeros(1, point_data.shape[1], dtype=torch.bool)
                    cir_used_objs[0, p_indices] = True
                    groups_used_objs = torch.cat([groups_used_objs, cir_used_objs], dim=0)

                    exist_combs += get_comb(p_indices, 3).tolist()
                    tensor_counter += 1
                    # print(f'circle group: {p_indices}')
        # print(f'\n')
        _, prob_indices = circle_tensor_candidates[:, config.group_tensor_index["circle"]].sort(descending=True)
        prob_indices = prob_indices[:args.e]
        circle_tensors[data_i] = circle_tensor_candidates[prob_indices]
        used_objs[data_i] = groups_used_objs[prob_indices]
    return circle_tensors, used_objs


def extend_line_group(args, obj_indices, obj_tensors):
    # points = obj_tensors[:, :, config.indices_position]
    has_new_element = True
    line_groups = copy.deepcopy(obj_tensors)
    line_group_indices = copy.deepcopy(obj_indices)

    while has_new_element:
        line_objs = obj_tensors[line_group_indices]
        extra_index = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(line_group_indices)))
        if len(extra_index) == 0:
            return None, None
        extra_objs = obj_tensors[extra_index]
        line_objs_extended = extend_groups(line_objs, extra_objs)
        colinearities = eval_utils.calc_colinearity(line_objs_extended, config.indices_position)
        avg_distances = eval_utils.calc_avg_dist(line_objs_extended, config.indices_position)
        is_line = colinearities < args.error_th
        is_even_dist = avg_distances < args.distribute_error_th
        passed_indices = is_line * is_even_dist
        has_new_element = passed_indices.sum() > 0
        passed_objs = extra_objs[passed_indices]
        passed_indices = extra_index[passed_indices]
        line_groups = torch.cat([line_objs, passed_objs], dim=0)
        line_group_indices += passed_indices.tolist()

    # check for evenly distribution
    # if not is_even_distributed_points(args, point_groups_new, shape="line"):
    #     return None, None
    line_group_indices = torch.tensor(line_group_indices)

    return line_groups, line_group_indices


def extend_circle_group(group_index, points, args):
    point_groups = points[group_index]
    c, r = eval_utils.predict_circles(point_groups, args.error_th)
    if c is None or r is None:
        return None, None, None, None
    extra_index = torch.tensor(sorted(set(list(range(points.shape[0]))) - set(group_index)))
    extra_points = points[extra_index]

    group_error = eval_utils.get_circle_error(c, r, extra_points)
    passed_points = extra_points[group_error < args.error_th]
    if len(passed_points) == 0:
        return None, None, None, None
    point_groups_extended = extend_groups(point_groups, passed_points)
    group_distribution = eval_utils.get_group_distribution(point_groups_extended, c)
    passed_indices = extra_index[group_error < args.error_th][group_distribution]
    passed_points = extra_points[group_error < args.error_th][group_distribution]
    point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    point_groups_indices_new = torch.cat([torch.tensor(group_index), passed_indices])
    # if not is_even_distributed_points(args, point_groups_new, shape="circle"):
    #     return None, None, None, None
    return point_groups_new, point_groups_indices_new, c, r


def extend_groups(group_objs, extend_objs):
    group_points_duplicate_all = group_objs.unsqueeze(0).repeat(extend_objs.shape[0], 1, 1)
    group_points_candidate = torch.cat([group_points_duplicate_all, extend_objs.unsqueeze(1)], dim=1)

    return group_points_candidate


def record_line_group(args, objs, obj_indices, img_i):
    line_tensor = aitk.utils.data_utils.to_line_tensor(objs).reshape(-1)
    # convert the group to a tensor

    # line_tensor_candidates = torch.cat([line_tensor_candidates, line_tensor], dim=0)

    # update point availabilities
    line_used_objs = torch.zeros(args.n_obj, dtype=torch.bool)
    line_used_objs[obj_indices] = True

    # groups_used_objs = torch.cat([groups_used_objs, line_used_objs], dim=0)
    print(f'(img {img_i}) line group: {obj_indices}')

    return line_tensor, line_used_objs


def detect_lines(args, obj_tensors):
    all_line_indices = []
    all_line_tensors = []

    # detect lines image by image
    for img_i in range(obj_tensors.shape[0]):
        line_indices_ith = []
        line_groups_ith = []
        exist_combs = []
        two_point_line_indices = get_comb(torch.tensor(range(obj_tensors[img_i].shape[0])), 2).tolist()
        # detect lines by checking each possible combinations
        for g_i, obj_indices in enumerate(two_point_line_indices):
            # check duplicate
            if obj_indices in exist_combs:
                continue
            line_obj_tensors, line_indices = extend_line_group(args, obj_indices, obj_tensors[img_i])

            if line_obj_tensors is not None and len(line_obj_tensors) >= args.line_group_min_sz:
                exist_combs += get_comb(line_indices, 2).tolist()

                line_indices_ith.append(line_indices)
                line_groups_ith.append(line_obj_tensors)
        all_line_indices.append(line_indices_ith)
        all_line_tensors.append(line_groups_ith)

    return all_line_indices, all_line_tensors


def encode_lines(args, percept_dict, obj_indices, obj_tensors):
    """
    Encode lines to tensors.
    """

    used_objs = torch.zeros(len(obj_tensors), args.group_e, args.n_obj, dtype=torch.bool)
    all_line_tensors = torch.zeros(len(obj_tensors), args.group_e, len(config.group_tensor_index.keys()))

    all_line_tensors = []
    all_line_tensors_indices = []

    for img_i in range(len(obj_tensors)):
        img_line_tensors = []
        img_line_tensors_indices = []
        for obj_i, objs in enumerate(obj_tensors[img_i]):
            line_tensor, line_used_objs = record_line_group(args, objs, obj_indices[img_i][obj_i], img_i)
            img_line_tensors.append(line_tensor.tolist())
            img_line_tensors_indices.append(line_used_objs.tolist())

        all_line_tensors.append(img_line_tensors)
        all_line_tensors_indices.append(img_line_tensors_indices)

    # all_line_tensors = torch.tensor(all_line_tensors)
    # all_line_tensors_indices = torch.tensor(all_line_tensors_indices)

    return all_line_tensors, all_line_tensors_indices


def prune_lines(args, lines_list, line_tensors_indices):
    tensor_len = len(config.group_tensor_index)
    n_obj = args.n_obj
    pruned_tensors = []
    pruned_tensors_indices = []
    for img_i in range(len(lines_list)):
        lines_ith_list = lines_list[img_i]

        img_scores = []

        if len(lines_ith_list) < args.group_max_e:
            d = args.group_max_e - len(lines_ith_list)
            lines_zero_list = [[0] * tensor_len] * d
            indices_zeros_list = [[False] * n_obj] * d
            lines_list[img_i] += lines_zero_list
            line_tensors_indices[img_i] += indices_zeros_list

        for line_tensor_list in lines_list[img_i]:
            line_tensor = torch.tensor(line_tensor_list)
            line_scores = line_tensor[config.group_tensor_index["line"]]
            img_scores.append(line_scores.tolist())
        # align tensor number to arg.group_e

        img_scores = torch.tensor(img_scores)
        _, prob_indices = img_scores.sort(descending=True)
        prob_indices = prob_indices[:args.group_max_e]

        if len(lines_list[img_i]) == 0:
            pruned_tensors.append(torch.zeros(size=(args.group_max_e, len(config.group_tensor_index))).tolist())
            pruned_tensors_indices.append(torch.zeros(args.n_obj, dtype=torch.bool).tolist())
        else:
            pruned_tensors.append(torch.tensor(lines_list[img_i])[prob_indices].tolist())
            pruned_tensors_indices.append(torch.tensor(line_tensors_indices[img_i])[prob_indices].tolist())

    return pruned_tensors, pruned_tensors_indices


def detect_dot_groups(args, percept_dict, valid_obj_indices):
    point_data = percept_dict[:, valid_obj_indices, config.indices_position]
    color_data = percept_dict[:, valid_obj_indices, config.indices_color]
    shape_data = percept_dict[:, valid_obj_indices, config.indices_shape]
    point_screen_data = percept_dict[:, valid_obj_indices, config.indices_screen_position]
    dot_tensors = torch.zeros(point_data.shape[0], args.e, len(config.group_tensor_index.keys()))
    for data_i in range(point_data.shape[0]):
        exist_combs = []
        tensor_counter = 0
        group_indices = get_comb(torch.tensor(range(point_data[data_i].shape[0])), 3).tolist()
        dot_tensor_candidates = torch.zeros(args.e, len(config.group_tensor_index.keys()))
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                p_groups, p_indices, center, r = extend_circle_group(group_index, point_data[data_i], args)
                if p_groups is not None and p_groups.shape[0] >= args.cir_group_min_sz:
                    colors = color_data[data_i][p_indices]
                    shapes = shape_data[data_i][p_indices]
                    p_groups_screen = point_screen_data[data_i][p_indices]
                    circle_tensor = to_circle_tensor(p_groups, p_groups_screen, colors, shapes, center=center,
                                                     r=r).reshape(1, -1)
                    dot_tensor_candidates = torch.cat([dot_tensor_candidates, circle_tensor], dim=0)
                    exist_combs += get_comb(p_indices, 3).tolist()
                    tensor_counter += 1
                    print(f'dot group: {p_indices}')
        print(f'\n')
        _, prob_indices = dot_tensor_candidates[:, config.group_tensor_index["circle"]].sort(descending=True)
        prob_indices = prob_indices[:args.e]
        dot_tensors[data_i] = dot_tensor_candidates[prob_indices]
    return dot_tensors


def detect_obj_groups(args, percept_dict_single, data_type):
    save_path = config.buffer_path / "hide" / args.dataset / "buffer_groups"
    save_file = save_path / f"{args.dataset}_group_res_{data_type}.pth.tar"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if os.path.exists(save_file):
        group_res = torch.load(save_file)
        line_tensors = group_res["tensors"]
        line_tensors_indices = group_res['used_objs']
    else:
        # TODO: improve the generalization. Using a more general function to detect multiple group shapes by taking
        #  specific detect functions as arguments
        line_tensors, line_tensors_indices = detect_line_groups(args, percept_dict_single, data_type)
        # pattern_dot = detect_dot_groups(args, percept_dict_single, used_objs)
        # pattern_cir, pattern_cir_used_objs = detect_circle_groups(args, percept_dict_single)
        # group_tensors, group_obj_index_tensors = merge_groups(args, line_tensors, pattern_cir, line_tensors_indices,
        #                                                       pattern_cir_used_objs)
        group_res = {"tensors": line_tensors, "used_objs": line_tensors_indices}
        torch.save(group_res, save_file)

    return line_tensors, line_tensors_indices,


def get_group_tree(args, obj_groups, group_by):
    """ root node corresponds the scene """
    """ Used for very complex scene """
    pos_groups, neg_groups = obj_groups[0], obj_groups[1]

    # cluster by color

    # cluster by shape

    # cluster by line and circle
    if args.neural_preds[group_by][0].name == 'group_shape':
        pos_groups = None

    return None


def test_groups(test_positive, test_negative, groups):
    accuracy = 0
    for i in range(test_positive.shape[0]):
        accuracy += test_groups_on_one_image(test_positive[i:i + 1], groups)
    for i in range(test_negative.shape[0]):
        neg_score = test_groups_on_one_image(test_negative[i:i + 1], groups)
        accuracy += 1 - neg_score
    accuracy = accuracy / (test_positive.shape[0] + test_negative.shape[0])
    print(f"test acc: {accuracy}")
    return accuracy


def test_groups_on_one_image(data, val_result):
    test_score = 1
    score_color_1, score_color_2 = eval_utils.eval_data(data[:, :, config.indices_color])
    eval_shape_positive_res, score_shape_1, score_shape_2 = eval_utils.eval_data(data[:, :, config.indices_shape])

    return test_score


def merge_groups(args, line_groups, cir_groups, line_used_objs, cir_used_objs):
    object_groups = torch.cat((line_groups, cir_groups), dim=1)
    used_objs = torch.cat((line_used_objs, cir_used_objs), dim=1)

    object_groups = object_groups[:, :args.group_e]
    used_objs = used_objs[:, :args.group_e]
    return object_groups, used_objs
