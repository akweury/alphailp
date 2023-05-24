import argparse
import os
import torch

import config
from data_utils import to_line_tensor, get_comb, to_circle_tensor
from eval_utils import eval_group_diff, eval_count_diff, count_func, group_func, get_circle_error, eval_score
from eval_utils import predict_circles, calc_colinearity, get_group_distribution
import eval_utils
import file_utils
import logic_utils
import log_utils
import copy


def detect_line_groups(args, percept_dict):
    point_data = percept_dict[:, :, config.indices_position]
    used_objs = torch.zeros(point_data.shape[0], args.group_e, point_data.shape[1], dtype=torch.bool)
    line_tensors = torch.zeros(point_data.shape[0], args.group_e, len(config.group_tensor_index.keys()))

    for data_i in range(point_data.shape[0]):
        exist_combs = []
        group_indices = get_comb(torch.tensor(range(point_data[data_i].shape[0])), 2).tolist()
        tensor_counter = 0
        line_tensor_candidates = torch.zeros(args.group_e, len(config.group_tensor_index.keys()))
        groups_used_objs = torch.zeros(args.group_e, point_data.shape[1], dtype=torch.bool)
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                point_groups, point_indices, collinearities = extend_line_group(args, group_index, point_data[data_i])
                if point_groups is not None and point_groups.shape[0] >= args.line_group_min_sz:
                    line_tensor = to_line_tensor(point_groups, percept_dict, data_i, point_indices).reshape(1, -1)
                    line_tensor_candidates = torch.cat([line_tensor_candidates, line_tensor], dim=0)

                    # update point availabilities
                    line_used_objs = torch.zeros(1, point_data.shape[1], dtype=torch.bool)
                    line_used_objs[0, point_indices] = True
                    groups_used_objs = torch.cat([groups_used_objs, line_used_objs], dim=0)
                    exist_combs += get_comb(point_indices, 2).tolist()
                    tensor_counter += 1
                    print(f'line group: {point_indices}')
        print(f'\n')
        group_scores = line_tensor_candidates[:, config.group_tensor_index["line"]]
        _, prob_indices = group_scores.sort(descending=True)
        prob_indices = prob_indices[:args.e]
        line_tensors[data_i] = line_tensor_candidates[prob_indices]
        used_objs[data_i] = groups_used_objs[prob_indices]
    return line_tensors, used_objs


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


def extend_line_group(args, group_index, points):
    has_new_element = True
    point_groups_new = copy.deepcopy(points)
    point_groups_indices_new = copy.deepcopy(group_index)
    colinearities = None

    while has_new_element:
        point_groups = points[point_groups_indices_new]
        extra_index = torch.tensor(sorted(set(list(range(points.shape[0]))) - set(point_groups_indices_new)))
        if len(extra_index) == 0:
            return None, None, None
        extra_points = points[extra_index]
        point_groups_extended = extend_groups(point_groups, extra_points)
        colinearities = calc_colinearity(point_groups_extended)
        avg_distances = eval_utils.calc_avg_dist(point_groups_extended)
        is_line = colinearities < args.error_th
        is_even_dist = avg_distances < args.distribute_error_th
        passed_indices = is_line * is_even_dist
        has_new_element = passed_indices.sum() > 0
        passed_points = extra_points[passed_indices]
        passed_indices = extra_index[passed_indices]
        point_groups_new = torch.cat([point_groups, passed_points], dim=0)
        point_groups_indices_new += passed_indices.tolist()

    # check for evenly distribution
    # if not is_even_distributed_points(args, point_groups_new, shape="line"):
    #     return None, None
    point_groups_indices_new = torch.tensor(point_groups_indices_new)

    return point_groups_new, point_groups_indices_new, colinearities


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


def extend_groups(point_groups, extend_points):
    group_points_duplicate_all = point_groups.unsqueeze(0).repeat(extend_points.shape[0], 1, 1)
    group_points_candidate = torch.cat([group_points_duplicate_all, extend_points.unsqueeze(1)], dim=1)
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--batch-size-train", type=int,
                        default=20, help="Batch size in nsfr train")
    parser.add_argument("--group_e", type=int, default=2,
                        help="The maximum number of object groups in one image")
    parser.add_argument("--e", type=int, default=5,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", default="red-triangle", help="Use kandinsky patterns dataset")
    parser.add_argument("--dataset-type", default="kandinsky",
                        help="kandinsky or clevr")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--with-pi", action="store_true",
                        help="Generate Clause with predicate invention.")
    parser.add_argument("--with-explain", action="store_true",
                        help="Explain Clause with predicate invention.")

    parser.add_argument("--small-data", action="store_true",
                        help="Use small training data.")
    parser.add_argument("--score_unique", action="store_false",
                        help="prune same score clauses.")
    parser.add_argument("--semantic_unique", action="store_false",
                        help="prune same semantic clauses.")
    parser.add_argument("--no-xil", action="store_true",
                        help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--small_data", action="store_false",
                        help="Use small portion of valuation data.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--min-beam", type=int, default=0,
                        help="The size of the minimum beam.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--cim-step", type=int, default=5,
                        help="The steps of clause infer module.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1,
                        help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=5,
                        help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101,
                        help="The number of epochs.")
    parser.add_argument("--pi_epochs", type=int, default=3,
                        help="The number of epochs for predicate invention.")
    parser.add_argument("--nc_max_step", type=int, default=3,
                        help="The number of max steps for nc searching.")
    parser.add_argument("--max_step", type=int, default=5,
                        help="The number of max steps for clause searching.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--suff_min", type=float, default=0.1,
                        help="The minimum accept threshold for sufficient clauses.")
    parser.add_argument("--sn_th", type=float, default=0.9,
                        help="The accept threshold for sufficient and necessary clauses.")
    parser.add_argument("--nc_th", type=float, default=0.9,
                        help="The accept threshold for necessary clauses.")
    parser.add_argument("--uc_th", type=float, default=0.8,
                        help="The accept threshold for unclassified clauses.")
    parser.add_argument("--sc_th", type=float, default=0.9,
                        help="The accept threshold for sufficient clauses.")
    parser.add_argument("--sn_min_th", type=float, default=0.2,
                        help="The accept sn threshold for sufficient or necessary clauses.")
    parser.add_argument("--similar_th", type=float, default=1e-3,
                        help="The minimum different requirement between any two clauses.")
    parser.add_argument("--semantic_th", type=float, default=0.75,
                        help="The minimum semantic different requirement between any two clauses.")
    parser.add_argument("--conflict_th", type=float, default=0.9,
                        help="The accept threshold for conflict clauses.")
    parser.add_argument("--length_weight", type=float, default=0.05,
                        help="The weight of clause length for clause evaluation.")
    parser.add_argument("--c_top", type=int, default=20,
                        help="The accept number for clauses.")
    parser.add_argument("--uc_good_top", type=int, default=10,
                        help="The accept number for unclassified good clauses.")
    parser.add_argument("--sc_good_top", type=int, default=20,
                        help="The accept number for sufficient good clauses.")
    parser.add_argument("--sc_top", type=int, default=20,
                        help="The accept number for sufficient clauses.")
    parser.add_argument("--nc_top", type=int, default=10,
                        help="The accept number for necessary clauses.")
    parser.add_argument("--nc_good_top", type=int, default=30,
                        help="The accept number for necessary good clauses.")
    parser.add_argument("--pi_top", type=int, default=20,
                        help="The accept number for pi on each classes.")
    parser.add_argument("--max_cluster_size", type=int, default=4,
                        help="The max size of clause cluster.")
    parser.add_argument("--min_cluster_size", type=int, default=2,
                        help="The min size of clause cluster.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("--top_data", type=int, default=20,
                        help="The maximum number of training data.")
    parser.add_argument("--with_bk", action="store_true",
                        help="Using background knowledge by PI.")
    parser.add_argument("--error_th", type=float, default=0.1,
                        help="The threshold for MAE of obj group fitting.")
    parser.add_argument("--line_group_min_sz", type=int, default=3,
                        help="The minimum objects allowed to form a line.")
    parser.add_argument("--cir_group_min_sz", type=int, default=5,
                        help="The minimum objects allowed to form a circle.")
    parser.add_argument("--group_conf_th", type=float, default=0.98,
                        help="The threshold of group confidence.")
    parser.add_argument("--maximum_obj_num", type=int, default=5,
                        help="The maximum number of objects/groups to deal with in a single image.")
    parser.add_argument("--distribute_error_th", type=float, default=0.3,
                        help="The threshold for group points forming a shape that evenly distributed on the whole shape.")
    args = parser.parse_args()

    args_file = config.data_path / "lang" / args.dataset_type / args.dataset / "args.json"
    file_utils.load_args_from_file(str(args_file), args)

    return args


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


def detect_obj_groups_single(args, percept_dict_single, data_type):
    group_path = config.buffer_path / "hide" / args.dataset / "buffer_groups"
    group_file = group_path / f"{args.dataset}_group_res_{data_type}.pth.tar"
    if not os.path.exists(group_path):
        os.mkdir(group_path)

    if os.path.exists(group_file):
        group_res = torch.load(group_file)
        group_tensors = group_res["tensors"]
        group_obj_index_tensors = group_res['used_objs']
    else:
        pattern_line, pattern_line_used_objs = detect_line_groups(args, percept_dict_single)
        # pattern_dot = detect_dot_groups(args, percept_dict_single, used_objs)
        pattern_cir, pattern_cir_used_objs = detect_circle_groups(args, percept_dict_single)
        group_tensors, group_obj_index_tensors = merge_groups(args, pattern_line, pattern_cir, pattern_line_used_objs,
                                                              pattern_cir_used_objs)
        group_res = {
            "tensors": group_tensors,
            "used_objs": group_obj_index_tensors,
        }
        torch.save(group_res, group_file)

    return group_tensors, group_obj_index_tensors


def detect_obj_groups_with_bk(args, percept_dict):

    return detect_res, obj_avail_res


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
