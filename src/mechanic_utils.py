import argparse

import torch

import config
from data_utils import to_line_tensor, get_comb, to_circle_tensor
from eval_utils import eval_group_diff, eval_count_diff, count_func, group_func, get_circle_error, eval_score
from eval_utils import predict_circles, predict_lines, get_group_distribution
from src import config, file_utils
from fol import bk

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
                    print(f'line group: {point_indices}')
        print(f'\n')
        _, prob_indices = line_tensor_candidates[:, config.group_tensor_index["line"]].sort(descending=True)
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
                p_groups, p_indices, center, r = extend_circle_group(group_index, data[data_i], args)
                if p_groups is not None and p_groups.shape[0] >= args.cir_group_min_sz:
                    circle_tensor = to_circle_tensor(p_groups, center=center, r=r).reshape(1, -1)
                    circle_tensor_candidates = torch.cat([circle_tensor_candidates, circle_tensor], dim=0)
                    exist_combs += get_comb(p_indices, 3).tolist()
                    tensor_counter += 1
                    print(f'circle group: {p_indices}')
        print(f'\n')
        _, prob_indices = circle_tensor_candidates[:, config.group_tensor_index["circle"]].sort(descending=True)
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


def extend_circle_group(group_index, points, args, ):
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
    passed_points = extra_points[passed_indices]
    point_groups_new = torch.cat([point_groups, passed_points], dim=0)
    point_groups_indices_new = torch.cat([torch.tensor(group_index), passed_indices])
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


def merge_groups(line_groups, cir_groups):
    object_groups = torch.cat((line_groups, cir_groups), dim=1)
    _, group_ranking = object_groups[:, :, -1].sort(dim=-1, descending=True)

    for i in range(object_groups.shape[0]):
        object_groups[i] = object_groups[i, group_ranking[i]]

    # group and groups suppose have to overlap with each other?


    return object_groups


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--batch-size-train", type=int,
                        default=20, help="Batch size in nsfr train")
    parser.add_argument("--group_e", type=int, default=4,
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
    parser.add_argument("--no-pi", action="store_true",
                        help="Generate Clause without predicate invention.")
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
    parser.add_argument("--cir_group_min_sz", type=int, default=4,
                        help="The minimum objects allowed to form a circle.")
    parser.add_argument("--group_conf_th", type=float, default=0.98,
                        help="The threshold of group confidence.")
    parser.add_argument("--maximum_obj_num", type=int, default=5,
                        help="The maximum number of objects/groups to deal with in a single image.")
    args = parser.parse_args()

    args_file = config.data_path / "lang" / args.dataset_type / args.dataset / "args.json"
    file_utils.load_args_from_file(str(args_file), args)

    return args


def update_args(args, pm_prediction_dict, obj_groups):
    args.val_pos = pm_prediction_dict["val_pos"].to(args.device)
    args.val_neg = pm_prediction_dict["val_neg"].to(args.device)
    args.group_pos = obj_groups[0]
    args.group_neg = obj_groups[1]
    args.data_size = args.val_pos.shape[0]
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset
    neural_preds = file_utils.load_neural_preds(bk.neural_predicate_3)
    args.neural_preds = [[neural_pred] for neural_pred in neural_preds]
    args.neural_preds.append(neural_preds)
    args.p_inv_counter = 0
