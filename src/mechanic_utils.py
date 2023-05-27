import os
import torch

import config
from data_utils import to_line_tensor, get_comb, to_circle_tensor
from eval_utils import eval_group_diff, eval_count_diff, count_func, group_func, get_circle_error, eval_score
from eval_utils import predict_circles, calc_colinearity, get_group_distribution
import eval_utils
import logic_utils
import log_utils
import copy


def record_line_group(args, objs, obj_indices):
    # convert the group to a tensor
    line_tensor = to_line_tensor(objs).reshape(1, -1)

    # line_tensor_candidates = torch.cat([line_tensor_candidates, line_tensor], dim=0)

    # update point availabilities
    line_used_objs = torch.zeros(1, args.n_obj, dtype=torch.bool)
    line_used_objs[0, obj_indices] = True

    # groups_used_objs = torch.cat([groups_used_objs, line_used_objs], dim=0)
    print(f'line group: {obj_indices}')

    return line_tensor, line_used_objs


def detect_lines(args, obj_tensors):
    all_line_indices = []
    all_line_tensors = []

    # detect lines image by image
    for data_i in range(obj_tensors.shape[0]):
        line_indices_ith = []
        line_groups_ith = []
        exist_combs = []
        two_point_line_indices = get_comb(torch.tensor(range(obj_tensors[data_i].shape[0])), 2).tolist()
        # detect lines by checking each possible combinations
        for g_i, obj_indices in enumerate(two_point_line_indices):
            # check duplicate
            if obj_indices in exist_combs:
                continue
            line_obj_tensors, line_indices = extend_line_group(args, obj_indices, obj_tensors[data_i])

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
            line_tensor, line_used_objs = record_line_group(args, objs, obj_indices[img_i][obj_i])
            img_line_tensors.append(line_tensor)
            img_line_tensors_indices.append(line_used_objs)

        all_line_tensors.append(img_line_tensors)
        all_line_tensors_indices.append(img_line_tensors_indices)

    return all_line_tensors, all_line_tensors_indices


def prune_lines(args, line_tensors, line_tensors_indices):
    pruned_tensors = []
    pruned_tensors_indices = []
    for img_i in range(len(line_tensors)):
        img_scores = []
        for line_tensor in line_tensors[img_i]:
            line_scores = line_tensor[0, config.group_tensor_index["line"]]
            img_scores.append(line_scores)

        img_scores = torch.tensor(img_scores)
        _, prob_indices = img_scores.sort(descending=True)
        prob_indices = prob_indices[:args.group_e]

        if len(line_tensors[img_i]) == 0:
            pruned_tensors.append(torch.zeros(size=(args.group_e, len(config.group_tensor_index))).tolist())
            pruned_tensors_indices.append(torch.zeros(1, args.n_obj, dtype=torch.bool).tolist())
        else:
            pruned_tensors.append(line_tensors[img_i][prob_indices].tolist())
            pruned_tensors_indices.append(line_tensors_indices[img_i][prob_indices].tolist())
    pruned_tensors_indices = torch.tensor(pruned_tensors_indices)
    pruned_tensors = torch.tensor(pruned_tensors)

    return pruned_tensors, pruned_tensors_indices


def detect_line_groups(args, obj_tensors):
    """
    input: object tensors
    output: group tensors
    """

    line_indices, line_groups = detect_lines(args, obj_tensors)
    line_tensors, line_tensors_indices = encode_lines(args, obj_tensors, line_indices, line_groups)

    # logic: prune strategy
    pruned_tensors, pruned_tensors_indices = prune_lines(args, line_tensors, line_tensors_indices)
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
            return None, None, None
        extra_objs = obj_tensors[extra_index]
        line_objs_extended = extend_groups(line_objs, extra_objs)
        colinearities = calc_colinearity(line_objs_extended)
        avg_distances = eval_utils.calc_avg_dist(line_objs_extended)
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
    c, r = predict_circles(point_groups, args.error_th)
    if c is None or r is None:
        return None, None, None, None
    extra_index = torch.tensor(sorted(set(list(range(points.shape[0]))) - set(group_index)))
    extra_points = points[extra_index]

    group_error = get_circle_error(c, r, extra_points)
    passed_points = extra_points[group_error < args.error_th]
    if len(passed_points) == 0:
        return None, None, None, None
    point_groups_extended = extend_groups(point_groups, passed_points)
    group_distribution = get_group_distribution(point_groups_extended, c)
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


def eval_single_group(data_pos, data_neg):
    group_pos_res, score_1, score_2 = eval_data(data_pos)
    group_neg_res, _, _ = eval_data(data_neg)
    res = eval_score(group_pos_res, group_neg_res)

    res_dict = {
        "result": res,
        "score_1": score_1,
        "score_2": score_2
    }

    return res_dict


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


def merge_groups(args, line_groups, cir_groups, line_used_objs, cir_used_objs):
    object_groups = torch.cat((line_groups, cir_groups), dim=1)
    used_objs = torch.cat((line_used_objs, cir_used_objs), dim=1)

    object_groups = object_groups[:, :args.group_e]
    used_objs = used_objs[:, :args.group_e]
    return object_groups, used_objs


# def clause_extension(pi_clauses, args, max_clause, lang, mode_declarations):
#     log_utils.add_lines(f"\n=== beam search iteration {args.iteration}/{args.max_step} ===", args.log_file)
#     index_pos = config.score_example_index["pos"]
#     index_neg = config.score_example_index["neg"]
#     eval_pred = ['kp']
#     clause_with_scores = []
#     # extend clauses
#     is_done = False
#     # if args.no_new_preds:
#     step = args.iteration
#     refs = args.last_refs
#     if args.pi_top == 0:
#         step = args.iteration
#         if len(args.last_refs) > 0:
#             refs = args.last_refs
#     while step <= args.iteration:
#         # log
#         log_utils.print_time(args, args.iteration, step, args.iteration)
#         # clause extension
#         refs_extended, is_done = extend_clauses(args, lang, mode_declarations, refs, pi_clauses)
#         if is_done:
#             break
#
#         self.NSFR = get_nsfr_model(args, self.lang, refs_extended, self.NSFR.atoms, pi_clauses, self.NSFR.fc)
#         # evaluate new clauses
#         score_all = eval_clause_infer.eval_clause_on_scenes(self.NSFR, args, eval_pred)
#         scores = eval_clause_infer.eval_clauses(score_all[:, :, index_pos], score_all[:, :, index_neg], args, step)
#         # classify clauses
#         clause_with_scores = eval_clause_infer.prune_low_score_clauses(refs_extended, score_all, scores, args)
#         # print best clauses that have been found...
#         clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)
#
#         new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
#         max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)
#
#         if args.pi_top > 0:
#             refs, clause_with_scores, is_done = prune_clauses(clause_with_scores, args)
#         else:
#             refs = logic_utils.top_select(clause_with_scores, args)
#         step += 1
#
#         if found_sn or len(refs) == 0:
#             is_done = True
#             break
#
#     args.is_done = is_done
#     args.last_refs = refs
#     return clause_with_scores, max_clause, step, args


def remove_conflict_clauses(refs, pi_clauses, args):
    # remove conflict clauses
    refs_non_conflict = logic_utils.remove_conflict_clauses(refs, pi_clauses, args)
    refs_non_trivial = logic_utils.remove_trivial_clauses(refs_non_conflict, args)

    log_utils.add_lines(f"after removing conflict clauses: {len(refs_non_trivial)} clauses left", args.log_file)
    return refs_non_trivial


def check_result(args, clause_with_scores, higher, max_clause, new_max_clause):
    if higher:
        best_clause = new_max_clause
    else:
        best_clause = max_clause

    if len(clause_with_scores) == 0:
        return best_clause, False
    elif clause_with_scores[0][1][2] == 1.0:
        return best_clause, True
    elif clause_with_scores[0][1][2] > args.sn_th:
        return best_clause, True
    return best_clause, False


def check_group_result(args, eval_res_val):
    shape_group_done = eval_res_val['shape_group']["result"] > args.group_conf_th
    color_done = eval_res_val['color']["result"] > args.group_conf_th
    shape_done = eval_res_val['shape']["result"] > args.group_conf_th

    is_done = shape_group_done.sum() + color_done.sum() + shape_done.sum() > 0

    return is_done


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
        line_tensors, line_tensors_indices = detect_line_groups(args, percept_dict_single)
        # pattern_dot = detect_dot_groups(args, percept_dict_single, used_objs)
        # pattern_cir, pattern_cir_used_objs = detect_circle_groups(args, percept_dict_single)
        # group_tensors, group_obj_index_tensors = merge_groups(args, line_tensors, pattern_cir, line_tensors_indices,
        #                                                       pattern_cir_used_objs)
        group_res = {
            "tensors": line_tensors,
            "used_objs": line_tensors_indices,
        }
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

# def get_lang_model(args, percept_dict, obj_groups):
#     clauses = []
#     # load language module
#     lang = Language(args, [])
# update language with neural predicate: shape/color/dir/dist


# PM = get_perception_module(args)
# VM = get_valuation_module(args, lang)
# PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
# FC = FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
#                                     pi_valuation_module=PI_VM, device=args.device)
# # Neuro-Symbolic Forward Reasoner for clause generation
# NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, pi_clauses, FC)
# PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)
#
# mode_declarations = get_mode_declarations(args, lang)
# clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, mode_declarations,
#                                    no_xil=args.no_xil)  # torch.device('cpu'))
#
# # pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang,
# #                                         no_xil=args.no_xil)  # torch.device('cpu'))


# def update_system(args, category, ):
#     # update arguments
#     clauses = []
#     p_inv_with_scores = []
#     # load language module
#     lang, vars, init_clauses, atoms = get_lang(args)
#     # update language with neural predicate: shape/color/dir/dist
#
#     if (category < len(args.neural_preds) - 1):
#         lang.preds = lang.preds[:2]
#         lang.invented_preds = []
#         lang.preds.append(args.neural_preds[category][0])
#         pi_clauses = []
#         pi_p = []
#     else:
#         print('last round')
#         lang.preds = lang.preds[:2] + args.neural_preds[-1]
#         lang.invented_preds = invented_preds
#         pi_clauses = all_pi_clauses
#         pi_p = invented_preds
#
#     atoms = logic_utils.get_atoms(lang)
#
#     args.is_done = False
#     args.iteration = 0
#     args.max_clause = [0.0, None]
#     args.no_new_preds = False
#     args.last_refs = init_clauses
#
#     PM = get_perception_module(args)
#     VM = get_valuation_module(args, lang)
#     PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
#     FC = facts_converter.FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
#                                         pi_valuation_module=PI_VM, device=args.device)
#     # Neuro-Symbolic Forward Reasoner for clause generation
#     NSFR_cgen = get_nsfr_model(args, lang, FC)
#     PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)
#
#     mode_declarations = get_mode_declarations(args, lang)
#     clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, mode_declarations,
#                                        no_xil=args.no_xil)  # torch.device('cpu'))
#
#     # pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang,
#     #                                         no_xil=args.no_xil)  # torch.device('cpu'))
#
#     return atoms, pi_clauses, pi_p
