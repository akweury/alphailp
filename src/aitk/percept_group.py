# Created by jing at 09.06.23
import copy
import os
import torch

import config

from aitk.utils import eval_utils
from aitk.utils import data_utils
from aitk.utils import visual_utils


def detect_groups(args, obj_tensors, g_type):
    if g_type == "line":
        init_obj_num = 2
        min_obj_num = args.line_group_min_sz
        extend_func = extend_line_group
    elif g_type == "cir":
        init_obj_num = 3
        min_obj_num = args.cir_group_min_sz
        extend_func = extend_circle_group
    elif g_type == "conic":
        init_obj_num = 5
        min_obj_num = args.conic_group_min_sz
        extend_func = extend_conic_group
    else:
        raise ValueError

    all_group_indices = []
    all_group_tensors = []
    all_group_shape_data = []
    all_group_fit_error = []

    # detect lines image by image
    for img_i in range(obj_tensors.shape[0]):
        group_indices_ith = []
        group_groups_ith = []
        group_shape_ith = []
        group_error_ith = []
        exist_combs = []
        min_group_indices = data_utils.get_comb(torch.tensor(range(obj_tensors[img_i].shape[0])), init_obj_num).tolist()
        # detect lines by checking each possible combinations
        for g_i, obj_indices in enumerate(min_group_indices):
            # check duplicate
            if obj_indices in exist_combs:
                continue
            g_indices, g_shape_data, fit_error = extend_func(args, obj_indices, obj_tensors[img_i])
            g_obj_tensors = obj_tensors[img_i][g_indices]
            if g_obj_tensors is not None and len(g_obj_tensors) >= min_obj_num:
                exist_combs += data_utils.get_comb(g_indices, init_obj_num).tolist()
                group_indices_ith.append(g_indices)
                group_groups_ith.append(g_obj_tensors)
                group_shape_ith.append(g_shape_data)
                group_error_ith.append(fit_error)
        all_group_indices.append(group_indices_ith)
        all_group_tensors.append(group_groups_ith)
        all_group_shape_data.append(group_shape_ith)
        all_group_fit_error.append(group_error_ith)
    return all_group_indices, all_group_tensors, all_group_shape_data, all_group_fit_error


def detect_cir(args, obj_tensors):
    point_data = obj_tensors[:, :, config.indices_position]
    color_data = obj_tensors[:, :, config.indices_color]
    shape_data = obj_tensors[:, :, config.indices_shape]
    point_screen_data = obj_tensors[:, :, config.indices_screen_position]

    used_objs = torch.zeros(point_data.shape[0], args.group_e, point_data.shape[1], dtype=torch.bool)
    circle_tensors = torch.zeros(point_data.shape[0], args.e, len(config.group_tensor_index.keys()))
    for data_i in range(point_data.shape[0]):
        exist_combs = []
        tensor_counter = 0
        group_indices = data_utils.get_comb(torch.tensor(range(point_data[data_i].shape[0])), 3).tolist()
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
                    circle_tensor = data_utils.to_circle_tensor(p_groups).reshape(1, -1)
                    circle_tensor_candidates = torch.cat([circle_tensor_candidates, circle_tensor], dim=0)

                    cir_used_objs = torch.zeros(1, point_data.shape[1], dtype=torch.bool)
                    cir_used_objs[0, p_indices] = True
                    groups_used_objs = torch.cat([groups_used_objs, cir_used_objs], dim=0)

                    exist_combs += data_utils.get_comb(p_indices, 3).tolist()
                    tensor_counter += 1
                    # print(f'circle group: {p_indices}')
        # print(f'\n')
        _, prob_indices = circle_tensor_candidates[:, config.group_tensor_index["circle"]].sort(descending=True)
        prob_indices = prob_indices[:args.e]
        circle_tensors[data_i] = circle_tensor_candidates[prob_indices]
        used_objs[data_i] = groups_used_objs[prob_indices]
    return circle_tensors, used_objs


def encode_groups(args, obj_indices, obj_tensors, group_type):
    """
    Encode groups to tensors.
    """
    obj_tensor_index = config.obj_tensor_index

    all_group_tensors = []
    all_group_tensors_indices = []

    for img_i in range(len(obj_tensors)):
        img_group_tensors = []
        img_group_tensors_indices = []
        for obj_i, objs in enumerate(obj_tensors[img_i]):
            group_used_objs = torch.zeros(args.n_obj, dtype=torch.bool)
            if group_type == "line":
                line_sc = eval_utils.fit_line(objs[:, [obj_tensor_index[i] for i in config.obj_screen_positions]])
                group_tensor = data_utils.to_line_tensor(objs, line_sc)
            elif group_type == "cir":
                cir = eval_utils.fit_circle(objs[:, [0, 2]], args)
                cir_sc = eval_utils.fit_circle(objs[:, [obj_tensor_index[i] for i in config.obj_screen_positions]],
                                               args)
                group_tensor = data_utils.to_circle_tensor(objs, cir, cir_sc)

            elif group_type == "conic":
                conics = eval_utils.fit_conic(objs[:, [0, 2]])
                conics_sc = eval_utils.fit_conic(objs[:, [obj_tensor_index[i] for i in config.obj_screen_positions]])
                group_tensor = data_utils.to_conic_tensor(objs, conics, conics_sc)
            else:
                raise ValueError

            group_used_objs[obj_indices[img_i][obj_i]] = True
            img_group_tensors.append(group_tensor.tolist())
            img_group_tensors_indices.append(group_used_objs.tolist())

        all_group_tensors.append(img_group_tensors)
        all_group_tensors_indices.append(img_group_tensors_indices)

    return all_group_tensors, all_group_tensors_indices


def prune_groups(args, g_tensors, group_tensors_indices, g_data, g_error):
    """ sort and select the high scoring groups"""
    pruned_tensors = []
    pruned_tensors_indices = []
    pruned_data = []
    pruned_error = []
    for img_i in range(len(g_tensors)):
        group_ith_list = g_tensors[img_i]
        img_scores = []
        if len(group_ith_list) < args.group_max_e:
            d = args.group_max_e - len(group_ith_list)
            g_tensors[img_i] += [[0] * len(config.group_tensor_index)] * d
            group_tensors_indices[img_i] += [[False] * args.n_obj] * d
            g_data[img_i] += [None] * d
            g_error[img_i] += [None] * d

        for group_tensor_list in g_tensors[img_i]:
            group_tensor = torch.tensor(group_tensor_list)
            group_scores = group_tensor[config.group_tensor_index["line"]]
            img_scores.append(group_scores.tolist())
        # align tensor number to arg.group_e
        img_scores = torch.tensor(img_scores)
        _, prob_indices = img_scores.sort(descending=True)
        prob_indices = prob_indices[:args.group_max_e]

        if len(g_tensors[img_i]) == 0:
            # no groups detected in the image
            pruned_tensors.append(torch.zeros(size=(args.group_max_e, len(config.group_tensor_index))).tolist())
            pruned_tensors_indices.append(torch.zeros(args.n_obj, dtype=torch.bool).tolist())
            pruned_data.append(None)
            pruned_error.append(None)
        else:
            pruned_tensors.append(torch.tensor(g_tensors[img_i])[prob_indices].tolist())
            pruned_tensors_indices.append(torch.tensor(group_tensors_indices[img_i])[prob_indices].tolist())
            pruned_data.append([g_data[img_i][i] for i in prob_indices.tolist()])
            pruned_error.append([g_error[img_i][i] for i in prob_indices.tolist()])

    return pruned_tensors, pruned_tensors_indices, pruned_data, pruned_error


def prune_cirs(args, cir_tensors, cir_tensors_indices):
    pass


def visual_group_analysis(args, g_tensors, g_indices, obj_tensors, g_type, g_shape_data, fit_error):
    # detect lines image by image
    for img_i in range(args.top_data):
        for g_i, obj_indices in enumerate(g_indices[img_i]):
            vis_file = args.analysis_output_path / f"{args.dataset}_img_{img_i}_{g_type}_g_{g_i}.png"
            data = g_shape_data[img_i][g_i]
            indices = g_indices[img_i][g_i]
            error = fit_error[img_i][g_i]
            objs = obj_tensors[img_i][indices]
            indices_rest = list(
                set(list(range(len(indices)))) - set([i for i, e in enumerate(indices) if e == True]))
            rest_objs = obj_tensors[img_i][indices_rest]
            if data is not None:
                visual_utils.visual_group(g_type, vis_file, data, objs, rest_objs, error)


def detect_line_groups(args, obj_tensors, data_type):
    """
    input: object tensors
    output: group tensors
    """

    line_indices, line_groups, line_data, line_error = detect_groups(args, obj_tensors, "line")
    line_tensors, line_tensors_indices = encode_groups(args, line_indices, line_groups, "line")
    # logic: prune strategy
    g_tensors, indices, data, error = prune_groups(args, line_tensors, line_tensors_indices, line_data, line_error)
    visual_group_analysis(args, g_tensors, indices, obj_tensors, "line", data, error)
    return g_tensors, indices


def detect_conic_groups(args, obj_tensors, data_type):
    """
    input: object tensors
    output: group tensors
    """

    conic_indices, conic_groups, conic_data, conic_error = detect_groups(args, obj_tensors, "conic")
    conic_tensors, conic_tensors_indices = encode_groups(args, conic_indices, conic_groups, "conic")
    tensors, indices, data, error = prune_groups(args, conic_tensors, conic_tensors_indices, conic_data, conic_error)
    visual_group_analysis(args, tensors, indices, obj_tensors, "conic", data, error)
    return tensors, indices


def detect_circle_groups(args, obj_tensors, data_type):
    """
    input: object tensors
    output: group tensors
    """

    cir_indices, cir_groups, cir_data, cir_error = detect_groups(args, obj_tensors, "cir")
    cir_tensors, cir_tensors_indices = encode_groups(args, cir_indices, cir_groups, "cir")
    tensors, indices, data, error = prune_groups(args, cir_tensors, cir_tensors_indices, cir_data, cir_error)
    visual_group_analysis(args, tensors, indices, obj_tensors, "cir", data, error)
    return tensors, indices


def extend_line_group(args, obj_indices, obj_tensors):
    colinearities = None
    # points = obj_tensors[:, :, config.indices_position]
    has_new_element = True
    line_groups = copy.deepcopy(obj_tensors)
    line_group_indices = copy.deepcopy(obj_indices)

    while has_new_element:
        line_objs = obj_tensors[line_group_indices]
        extra_index = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(line_group_indices)))
        if len(extra_index) == 0:
            break
        extra_objs = obj_tensors[extra_index]
        line_objs_extended = extend_groups(line_objs, extra_objs)
        tensor_index = config.group_tensor_index
        colinearities = eval_utils.calc_colinearity(line_objs_extended,
                                                    [tensor_index[i] for i in config.obj_positions])
        avg_distances = eval_utils.calc_avg_dist(line_objs_extended, [tensor_index[i] for i in config.obj_positions])

        is_line = colinearities < args.error_th
        is_even_dist = avg_distances < args.distribute_error_th
        passed_indices = is_line * is_even_dist
        has_new_element = passed_indices.sum() > 0
        passed_objs = extra_objs[passed_indices]
        passed_indices = extra_index[passed_indices]
        line_groups = torch.cat([line_objs, passed_objs], dim=0)
        line_group_indices += passed_indices.tolist()

    line_group_indices = torch.tensor(line_group_indices)
    line = eval_utils.fit_line(obj_tensors[line_group_indices][:, [0, 2]])

    return line_group_indices, line, colinearities


def extend_conic_group(args, obj_indices, obj_tensors):
    poly_error = None
    has_new_element = True
    poly_group_indices = copy.deepcopy(obj_indices)

    group_objs = obj_tensors[poly_group_indices]
    conics = eval_utils.fit_conic(group_objs[:, [0, 2]])
    if conics["axis"][0] <= 0 or conics["axis"][1] <= 0:
        group_objs = None
        conics = None

    while has_new_element and conics is not None:

        leaf_indices = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(poly_group_indices)))
        if len(leaf_indices) == 0:
            break
        leaf_objs = obj_tensors[leaf_indices]
        all_points = [group_objs, leaf_objs]

        poly_error = eval_utils.get_conic_error(conics["coef"], conics["center"], leaf_objs[:, [0, 2]])

        is_poly = poly_error < args.poly_error_th
        has_new_element = is_poly.sum() > 0
        passed_leaf_objs = leaf_objs[is_poly]
        passed_leaf_indices = leaf_indices[is_poly]
        group_objs = torch.cat([group_objs, passed_leaf_objs], dim=0)
        poly_group_indices += passed_leaf_indices.tolist()

    poly_group_indices = torch.tensor(poly_group_indices)

    return poly_group_indices, conics, poly_error


def extend_circle_group(args, obj_indices, obj_tensors):
    cir_error = None
    has_new_element = True
    # cir_groups = copy.deepcopy(obj_tensors)
    cir_group_indices = copy.deepcopy(obj_indices)

    group_objs = obj_tensors[cir_group_indices]
    cir = eval_utils.fit_circle(group_objs[:, [0, 2]], args)

    while has_new_element and cir is not None:

        leaf_indices = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(cir_group_indices)))
        if len(leaf_indices) == 0:
            break
        leaf_objs = obj_tensors[leaf_indices]
        # branch_objs = extend_groups(seed_objs, leaf_objs)

        cir_error = eval_utils.get_circle_error(cir["center"], cir["radius"], leaf_objs[:, [0, 2]])

        is_circle = cir_error < args.cir_error_th
        has_new_element = is_circle.sum() > 0
        passed_leaf_objs = leaf_objs[is_circle]
        passed_leaf_indices = leaf_indices[is_circle]
        group_objs = torch.cat([group_objs, passed_leaf_objs], dim=0)
        cir_group_indices += passed_leaf_indices.tolist()
    cir_group_indices = torch.tensor(cir_group_indices)

    return cir_group_indices, cir, cir_error

    # point_groups = obj_tensors[obj_indices]
    # c, r = eval_utils.calc_circles(point_groups, args.error_th)
    # if c is None or r is None:
    #     return None, None
    # extra_index = torch.tensor(sorted(set(list(range(obj_tensors.shape[0]))) - set(obj_indices)))
    # extra_points = obj_tensors[extra_index]
    #
    # group_error = eval_utils.get_circle_error(c, r, extra_points)
    # passed_points = extra_points[group_error < args.error_th]
    # if len(passed_points) == 0:
    #     return None, None
    # point_groups_extended = extend_groups(point_groups, passed_points)
    # group_distribution = eval_utils.get_group_distribution(point_groups_extended, c)
    # passed_indices = extra_index[group_error < args.error_th][group_distribution]
    # passed_points = extra_points[group_error < args.error_th][group_distribution]
    # point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    # point_groups_indices_new = torch.cat([torch.tensor(obj_indices), passed_indices])
    # # if not is_even_distributed_points(args, point_groups_new, shape="circle"):
    # #     return None, None, None, None
    # cir_data = {
    #     "groups": point_groups_new,
    #     "centers": c,
    #     "radius": r
    # }
    # return cir_data, point_groups_indices_new


def extend_groups(group_objs, extend_objs):
    group_points_duplicate_all = group_objs.unsqueeze(0).repeat(extend_objs.shape[0], 1, 1)
    group_points_candidate = torch.cat([group_points_duplicate_all, extend_objs.unsqueeze(1)], dim=1)

    return group_points_candidate


def detect_dot_groups(args, percept_dict, valid_obj_indices):
    point_data = percept_dict[:, valid_obj_indices, config.indices_position]
    color_data = percept_dict[:, valid_obj_indices, config.indices_color]
    shape_data = percept_dict[:, valid_obj_indices, config.indices_shape]
    point_screen_data = percept_dict[:, valid_obj_indices, config.indices_screen_position]
    dot_tensors = torch.zeros(point_data.shape[0], args.e, len(config.group_tensor_index.keys()))
    for data_i in range(point_data.shape[0]):
        exist_combs = []
        tensor_counter = 0
        group_indices = data_utils.get_comb(torch.tensor(range(point_data[data_i].shape[0])), 3).tolist()
        dot_tensor_candidates = torch.zeros(args.e, len(config.group_tensor_index.keys()))
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                p_groups, p_indices, center, r = extend_circle_group(group_index, point_data[data_i], args)
                if p_groups is not None and p_groups.shape[0] >= args.cir_group_min_sz:
                    colors = color_data[data_i][p_indices]
                    shapes = shape_data[data_i][p_indices]
                    p_groups_screen = point_screen_data[data_i][p_indices]
                    circle_tensor = data_utils.to_circle_tensor(p_groups).reshape(1, -1)
                    dot_tensor_candidates = torch.cat([dot_tensor_candidates, circle_tensor], dim=0)
                    exist_combs += data_utils.get_comb(p_indices, 3).tolist()
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
    args.analysis_output_path = args.analysis_path / data_type

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if os.path.exists(save_file):
        group_res = torch.load(save_file)
        group_tensors = group_res["tensors"]
        group_obj_index_tensors = group_res['used_objs']
    else:
        line_tensors, line_tensors_indices = detect_line_groups(args, percept_dict_single, data_type)
        conic_tensors, conic_tensors_indices = detect_conic_groups(args, percept_dict_single, data_type)
        cir_tensors, cir_tensors_indices = detect_circle_groups(args, percept_dict_single, data_type)

        group_tensors, group_obj_index_tensors = merge_groups(args, line_tensors, cir_tensors, conic_tensors,
                                                              line_tensors_indices, cir_tensors_indices,
                                                              conic_tensors_indices)
        visual_utils.visual_groups(args, group_tensors, percept_dict_single, group_obj_index_tensors, data_type)
        group_res = {"tensors": group_tensors, "used_objs": group_obj_index_tensors}
        torch.save(group_res, save_file)

    return group_tensors, group_obj_index_tensors,


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


def select_top_k_groups(args, object_groups, used_objs):
    obj_groups_top = []
    obj_groups_top_indices = []
    group_indices = data_utils.get_comb(torch.tensor(range(object_groups.shape[1])), args.group_e)
    for img_i in range(args.top_data):
        groups = object_groups[img_i]
        # select groups including as many objects as possible
        objs_max_selection = group_indices[0]
        used_objs_max = 0
        for group_index in group_indices:
            comb_used_objs = torch.sum(used_objs[img_i, group_index.tolist(), :], dim=0)
            obj_num = data_utils.op_count_nonzeros(comb_used_objs, axis=0, epsilon=1e-10)
            if obj_num > used_objs_max:
                objs_max_selection = group_index
                used_objs_max = obj_num
        obj_groups_img_top = groups[objs_max_selection.tolist()[:args.group_e]]
        obj_groups_img_top_indices = used_objs[img_i, objs_max_selection.tolist()[:args.group_e]]

        obj_groups_top.append(obj_groups_img_top.tolist())
        obj_groups_top_indices.append(obj_groups_img_top_indices.tolist())

        # log
        for g_i, obj_group in enumerate(obj_groups_img_top):
            if obj_group[config.group_tensor_index["circle"]] > 0.9:
                group_name = "circle"
                group_msg = f""
            elif obj_group[config.group_tensor_index["line"]] > 0.9:
                group_name = "line"
                group_msg = f""
            elif obj_group[config.group_tensor_index["conic"]] > 0.9:
                group_name = "conic"
                axis_a = obj_group[config.group_tensor_index["screen_axis_x"]]
                axis_b = obj_group[config.group_tensor_index["screen_axis_z"]]
                group_msg = f"axis a: {axis_a}, axis b: {axis_b}"
            else:
                group_name = "unknown"
                group_msg = f""
            group_obj_indices = ((obj_groups_img_top_indices[g_i] == True).nonzero(as_tuple=True)[0])
            print(f'(img {img_i}) {group_name} {group_obj_indices} {group_msg}')

    return obj_groups_top, obj_groups_top_indices


def merge_groups(args, line_groups, cir_groups, conic_groups, line_used_objs, cir_used_objs, conic_used_objs):
    line_groups = torch.tensor(line_groups)
    cir_groups = torch.tensor(cir_groups)
    conic_groups = torch.tensor(conic_groups)

    line_used_objs = torch.tensor(line_used_objs)
    cir_used_objs = torch.tensor(cir_used_objs)
    conic_used_objs = torch.tensor(conic_used_objs)

    object_groups = torch.cat((line_groups, cir_groups, conic_groups), dim=1)
    used_objs = torch.cat((line_used_objs, cir_used_objs, conic_used_objs), dim=1)

    # select top-k groups
    top_k_groups, top_k_group_indices = select_top_k_groups(args, object_groups, used_objs)

    return top_k_groups, top_k_group_indices
