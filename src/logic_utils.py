import numpy as np
import torch
from infer import InferModule, ClauseInferModule
from eval_clause_infer import eval_clause_sign
from tensor_encoder import TensorEncoder
from fol.logic import *
from fol.data_utils import DataUtils
from fol.language import DataType
import glob

import config
import log_utils
import eval_clause_infer

p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])


def get_lang(args):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """

    du = DataUtils(lark_path=args.lark_path, lang_base_path=args.lang_base_path,
                   dataset_type=args.dataset_type, dataset=args.dataset)
    lang = du.load_language(args)
    init_clauses = du.load_clauses(str(du.base_path / 'clauses.txt'), lang, args)
    atoms = generate_atoms(lang)
    vars = [Var(f"O{i + 1}") for i in range(args.e)]
    return lang, vars, init_clauses, atoms


def get_atoms(lang):
    atoms = generate_atoms(lang)
    return atoms


def build_infer_module(args, clauses, pi_clauses, atoms, lang, device, m=3, infer_step=3, train=False, gamma=None):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    te_bk = None
    I_bk = None
    if len(pi_clauses) > 0:
        te_pi = TensorEncoder(lang, atoms, pi_clauses, device=device)
        I_pi = te_pi.encode()
    else:
        te_pi = None
        I_pi = None
    ##I_bk = None
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk, I_pi=I_pi, gamma=gamma)
    return im


def build_clause_infer_module(args, clauses, pi_clauses, atoms, lang, device, m=3, infer_step=5, train=False,
                              gamma=None):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    te_bk = None
    I_bk = None

    te_pi = None
    I_pi = None
    if len(pi_clauses) > 0:
        te_pi = TensorEncoder(lang, atoms, pi_clauses, device=device)
        I_pi = te_pi.encode()

    im = ClauseInferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk, I_pi=I_pi, gamma=gamma)
    return im


def build_pi_clause_infer_module(args, clauses, pi_clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()

    te_bk = None
    I_bk = None

    te_pi = None
    I_pi = None
    if len(pi_clauses) > 0:
        te_pi = TensorEncoder(lang, atoms, pi_clauses, device=device)
        I_pi = te_pi.encode()

    im = ClauseInferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk, I_pi=I_pi)
    return im


def generate_atoms(lang):
    spec_atoms = [false, true]
    atoms = []
    for pred in lang.preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))
        # args_list = lang.get_args_by_pred(pred)
        args_str_list = []
        # args_mem = []
        for args in args_list:

            # check if args and pred correspond are in the same area
            if pred.dtypes[0].name == 'area':
                if pred.name[0] + pred.name[5:] != args[0].name:
                    continue

            if len(args) == 1 or len(set(args)) == len(args):
                # if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                # if len(set(args)) == len(args):
                # if not (str(sorted([str(arg) for arg in args])) in args_str_list):
                atoms.append(Atom(pred, args))
                # args_str_list.append(
                #    str(sorted([str(arg) for arg in args])))
                # print('add atom: ', Atom(pred, args))
    pi_atoms = []
    for pred in lang.invented_preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))

        args_str_list = []

        for args in args_list:
            if len(args) == 1 or len(set(args)) == len(args):
                pi_atoms.append(Atom(pred, args))
    bk_pi_atoms = []
    for pred in lang.bk_inv_preds:
        dtypes = pred.dtypes
        consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*consts_list)))
        for args in args_list:
            # check if args and pred correspond are in the same area
            if pred.dtypes[0].name == 'area':
                if pred.name[0] + pred.name[5:] != args[0].name:
                    continue
            if len(args) == 1 or len(set(args)) == len(args):
                pi_atoms.append(Atom(pred, args))
    return spec_atoms + sorted(atoms) + sorted(pi_atoms) + sorted(bk_pi_atoms)


def generate_bk(lang):
    atoms = []
    for pred in lang.preds:
        if pred.name in ['diff_color', 'diff_shape']:
            dtypes = pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = itertools.product(*consts_list)
            for args in args_list:
                if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                    atoms.append(Atom(pred, args))
    return atoms


def get_index_by_predname(pred_str, atoms):
    indices = []
    for p_i, p_str in enumerate(pred_str):
        p_indices = []
        for i, atom in enumerate(atoms):
            if atom.pred.name == p_str:
                p_indices.append(i)
        indices.append(p_indices)
    return indices


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]


def get_pi_bodies_by_name(pi_clauses, pi_name):
    pi_bodies_all = []
    # name_change_dict = {"A": "O1", "B": "O2", "X": "X", "O2": "O2", "O1": "O1"}
    for pi_c in pi_clauses:
        if pi_name == pi_c.head.pred.name:
            pi_bodies = []
            for b in pi_c.body:
                p_name = b.pred.name
                if "inv_pred" in p_name:
                    body_names = get_pi_bodies_by_name(pi_clauses, p_name)
                    pi_bodies += body_names
                else:
                    # b.terms[0].name = name_change_dict[b.terms[0].name]
                    # b.terms[1].name = name_change_dict[b.terms[1].name]
                    pi_bodies.append(b)

            pi_bodies_all += pi_bodies

    return pi_bodies_all


def change_pi_body_names(pi_clauses):
    name_change_dict = {"A": "O1", "B": "O2", "X": "X"}
    for pi_c in pi_clauses:
        pi_c.head.terms[0] = name_change_dict[pi_c.head.terms[0].name]
        if len(pi_c.head.terms) > 1:
            pi_c.head.terms[1] = name_change_dict[pi_c.head.terms[1].name]
        for b in pi_c.body:
            b.terms[0].name = name_change_dict[b.terms[0].name]
            if len(b.terms) > 1:
                b.terms[1].name = name_change_dict[b.terms[1].name]

    return pi_clauses


def remove_conflict_clauses(clauses, pi_clauses, args):
    # print("\nCheck for conflict clauses...")
    clause_ordered = []
    non_conflict_clauses = []
    for clause in clauses:
        is_conflict = False
        with_pi = False
        if len(pi_clauses) > 0:
            for cb in clause.body:
                if "inv_pred" in cb.pred.name:
                    with_pi = True
            if not with_pi:
                is_conflict = False
        if with_pi or len(pi_clauses) == 0:
            for i in range(len(clause.body)):
                if is_conflict:
                    break
                for j in range(len(clause.body)):
                    if i == j:
                        continue
                    if "at_area" in clause.body[i].pred.name and "at_area" in clause.body[j].pred.name:
                        if clause.body[i].terms == clause.body[j].terms:
                            is_conflict = True
                            break
                            # print(f'conflict clause: {clause}')

                        elif conflict_pred(clause.body[i].pred.name,
                                           clause.body[j].pred.name,
                                           list(clause.body[i].terms),
                                           list(clause.body[j].terms)):
                            is_conflict = True
                            break
                            # print(f'conflict clause: {clause}')

                    if "inv_pred" in clause.body[j].pred.name and not is_conflict:
                        pi_name = clause.body[j].pred.name
                        pi_bodies = get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
                    if "inv_pred" in clause.body[i].pred.name and not is_conflict:
                        pi_name = clause.body[i].pred.name
                        pi_bodies = get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
                    # if "at_are_6" in clause.body[i].pred.name or "at_are_6" in clause.body[j].pred.name:
                    #     print("conflict")

        if not is_conflict:
            non_conflict_clauses.append(clause)
        # else:
        #     log_utils.add_lines(f"(conflict clause) {clause}", args.log_file)

    return non_conflict_clauses


def conflict_pred(p1, p2, t1, t2):
    non_confliect_dict = {
        "at_area_0": ["at_area_2"],
        "at_area_1": ["at_area_3"],
        "at_area_2": ["at_area_0"],
        "at_area_3": ["at_area_1"],
        "at_area_4": ["at_area_6"],
        "at_area_5": ["at_area_7"],
        "at_area_6": ["at_area_4"],
        "at_area_7": ["at_area_5"],
    }
    if p1 in non_confliect_dict.keys():
        if "at_area" in p2 and p2 not in non_confliect_dict[p1]:
            if t1[0] == t2[1] and t2[0] == t1[1]:
                return True
    return False


def is_repeat_clu(clu, clu_list):
    is_repeat = False
    for ref_clu, clu_score in clu_list:
        if clu == ref_clu:
            is_repeat = True
            break
    return is_repeat


def has_same_preds(c1, c2):
    if len(c1.body) != len(c2.body):
        return False
    same_preds = True
    for i in range(len(c1.body)):
        if not c1.body[i].pred.name == c2.body[i].pred.name:
            same_preds = False
    if same_preds:
        return True
    else:
        return False


def has_same_preds_and_atoms(c1, c2):
    if len(c1.body) != len(c2.body):
        return False
    same_preds = True
    for i in range(len(c1.body)):
        if not same_preds:
            break
        if not c1.body[i].pred.name == c2.body[i].pred.name:
            same_preds = False
        else:
            for j, term in enumerate(c1.body[i].terms):
                if "O" not in term.name:
                    if not term.name == c2.body[i].terms[j].name:
                        same_preds = False
    if same_preds:
        return True
    else:
        return False


def check_trivial_clusters(clause_clusters):
    clause_clusters_untrivial = []
    for c_clu in clause_clusters:
        is_trivial = False
        if len(c_clu) > 1:
            for c_i, c in enumerate(c_clu):
                clause = c[1]
                clause.body = sorted(clause.body)
                if c_i > 0:
                    if has_same_preds_and_atoms(clause, c_clu[0][1]):
                        is_trivial = True
                        break
                    if not has_same_preds(clause, c_clu[0][1]):
                        is_trivial = True
                        break
        if not is_trivial:
            clause_clusters_untrivial.append(c_clu)
    return clause_clusters_untrivial


def get_independent_clusters(clauses, args):
    print(f"\nsearching for independent clauses from {len(clauses)} clauses...")

    clauses_with_score = []
    for clause_i, [clause, four_scores, c_scores] in enumerate(clauses):
        clauses_with_score.append([clause_i, clause, c_scores])

    clause_clusters = sub_lists(clauses_with_score, min_len=args.min_cluster_size, max_len=args.max_cluster_size)

    return clause_clusters


def search_independent_clauses_parallel(clauses, total_score, args):
    patterns = get_independent_clusters(clauses, args)
    # trivial: contain multiple semantic identity bodies
    patterns = check_trivial_clusters(patterns)

    # necessary_clusters = []
    # sufficient_clusters = []
    # sn_clusters = []
    # sn_th_clusters = []
    # nc_th_clusters = []
    # sc_th_clusters = []
    # other_clusters = []
    # TODO: parallel programming
    index_neg = config.score_example_index["neg"]
    index_pos = config.score_example_index["pos"]

    # evaluate each new pattern
    clu_all = []
    for cc_i, pattern in enumerate(patterns):
        score_neg = torch.zeros((1, total_score, len(pattern))).to(args.device)
        score_pos = torch.zeros((1, total_score, len(pattern))).to(args.device)
        # score_max = torch.zeros(size=(score_neg.shape[0], score_neg.shape[1], 2)).to(args.device)

        for f_i, [c_i, c, c_score] in enumerate(pattern):
            score_neg[0, :, f_i] = c_score[:, index_neg]
            score_pos[0, :, f_i] = c_score[:, index_pos]

        score_neg = score_neg.max(dim=2, keepdims=True)[0]
        score_pos = score_pos.max(dim=2, keepdims=True)[0]

        # score_max[:, :, index_pos] = score_pos[:, :, 0]
        # score_max[:, :, index_neg] = score_neg[:, :, 0]

        score_all = eval_clause_infer.eval_clauses(score_pos, score_neg, args, len(pattern[0][1].body) - args.e)
        clu_all.append([pattern, score_all])

    index_suff = config.score_type_index['suff']
    index_ness = config.score_type_index['ness']
    index_sn = config.score_type_index['sn']

    clu_suff = [clu for clu in clu_all if clu[1][index_suff] > args.sc_th and clu[1][index_sn] > args.sn_min_th]
    clu_ness = [clu for clu in clu_all if clu[1][index_ness] > args.nc_th and clu[1][index_sn] > args.sn_min_th]
    clu_sn = [clu for clu in clu_all if clu[1][index_sn] > args.sn_th]
    clu_classified = sorted(clu_suff + clu_ness + clu_sn, key=lambda x: x[1][2], reverse=True)
    clu_lists_sorted = sorted(clu_all, key=lambda x: x[1][index_ness], reverse=True)
    return clu_classified


def sub_clause_of(clause_a, clause_b):
    """
    Check if clause a is a sub-clause of clause b
    Args:
        clause_a:
        clause_b:

    Returns:

    """
    for body_a in clause_a.body:
        if body_a not in clause_b.body:
            return False

    return True


def sub_lists(l, min_len=0, max_len=None):
    # initializing empty list
    comb = []

    # Iterating till length of list
    if max_len is None:
        max_len = len(l) + 1
    for i in range(min_len, max_len):
        # Generating sub list
        comb += [list(j) for j in itertools.combinations(l, i)]
    # Returning list
    return comb


def eval_clause_clusters(clause_clusters, clause_scores_full):
    """
    Scoring each clause cluster, ranking them, return them.
    Args:
        clause_clusters:
        clause_scores_full:

    Returns:

    """
    cluster_candidates = []
    for c_index, clause_cluster in enumerate(clause_clusters):
        clause = list(clause_cluster.keys())[0]
        clause_full_score = clause_scores_full[c_index]
        total_score = np.sum(clause_full_score)
        complementary_clauses = [list(ic_dict.keys())[0] for ic_dict in list(clause_cluster.values())[0]]
        complementary_clauses_index = [list(ic_dict.values())[0] for ic_dict in list(clause_cluster.values())[0]]
        complementary_clauses_full_score = [clause_scores_full[index] for index in complementary_clauses_index]

        complementary_clauses_full_score_no_repeat = []
        complementary_clauses_no_repeat = []
        for i, full_scores_i in enumerate(complementary_clauses_full_score):
            for j, full_scores_j in enumerate(complementary_clauses_full_score):
                if i == j:
                    continue
                if full_scores_j == full_scores_i:
                    if full_scores_j in complementary_clauses_full_score_no_repeat:
                        continue
                    else:
                        complementary_clauses_full_score_no_repeat.append(full_scores_j)
                        complementary_clauses_no_repeat.append(complementary_clauses[j])

        sum_pos_clause = clause_full_score[1] + clause_full_score[3]
        sum_pos_clause_ind = [fs[1] + fs[3] for fs in complementary_clauses_full_score_no_repeat]

        if (sum_pos_clause + np.sum(sum_pos_clause_ind)) == total_score:
            cluster_candidates.append(clause_cluster)
        elif (sum_pos_clause + np.sum(sum_pos_clause_ind)) >= total_score:
            sum_ind = total_score - sum_pos_clause
            # find a subset of sum_pos_clause_ind, so that the sum of the subset equal to sum_ind
            sub_sets_candidates = []
            subsets = []
            for i in range(0, len(sum_pos_clause_ind) + 1):  # to get all lengths: 0 to 3
                for subset in itertools.combinations(sum_pos_clause_ind, i):
                    subsets.append(subset)
            sum_pos_clause_ind_index = [i for i in range(len(sum_pos_clause_ind))]
            subset_index = []
            for i in range(0, len(sum_pos_clause_ind_index) + 1):  # to get all lengths: 0 to 3
                for subset in itertools.combinations(sum_pos_clause_ind_index, i):
                    subset_index.append(subset)

            for subset_i, subset in enumerate(subsets):
                if np.sum(list(subset)) == sum_ind:
                    indices = subset_index[subset_i]
                    clauses_set = [clause_cluster[clause][c_i] for c_i in indices]
                    cluster_candidates.append({
                        "clause": clause, "clause_score": sum_pos_clause, "clause_set": clauses_set,
                        "clause_set_score": subset,
                    })
    new_pi_clauses = []

    for cluster in cluster_candidates:
        cluster_ind = [list(c.keys())[0] for c in cluster["clause_set"]]
        new_pi_clauses.append([cluster["clause"]] + cluster_ind)
    return new_pi_clauses


def get_four_scores(predicate_scores):
    return eval_clause_sign(predicate_scores)[0][1]


def eval_predicates_slow(NSFR, args, pred_names, pos_pred, neg_pred):
    bz = args.batch_size
    device = args.device
    pos_img_num = pos_pred.shape[0]
    neg_img_num = neg_pred.shape[0]
    eval_pred_num = len(pred_names)
    clause_num = len(NSFR.clauses)
    score_positive = torch.zeros((bz, pos_img_num, clause_num, eval_pred_num)).to(device)
    score_negative = torch.zeros((bz, neg_img_num, clause_num, eval_pred_num)).to(device)
    # get predicates that need to be evaluated.
    # pred_names = ['kp']
    # for pi_c in pi_clauses:
    #     for body_atom in pi_c.body:
    #         if "inv_pred" in body_atom.pred.name:
    #             pred_names.append(body_atom.pred.name)

    for image_index in range(pos_img_num):
        V_T_list = NSFR.clause_eval_quick(pos_pred[image_index].unsqueeze(0)).detach()
        A = V_T_list.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG

        C_score = torch.zeros((bz, clause_num, eval_pred_num)).to(device)
        # clause loop
        for clause_index, V_T in enumerate(V_T_list):
            for pred_index, pred_name in enumerate(pred_names):
                predicted = NSFR.ilp_predict(v=V_T_list[clause_index, 0:1, :], predname=pred_name).detach()
                C_score[:, clause_index, pred_index] = predicted
        # sum over positive prob
        score_positive[:, image_index, :] = C_score

    # negative image loop
    for image_index in range(neg_img_num):
        V_T_list = NSFR.clause_eval_quick(neg_pred[image_index].unsqueeze(0)).detach()

        C_score = torch.zeros((bz, clause_num, eval_pred_num)).to(device)
        for clause_index, V_T in enumerate(V_T_list):
            for pred_index, pred_name in enumerate(pred_names):
                predicted = NSFR.ilp_predict(v=V_T_list[clause_index, 0:1, :], predname=pred_name).detach()
                C_score[:, clause_index, pred_index] = predicted
            # C
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
        score_negative[:, image_index, :] = C_score

    # axis: batch_size, pred_names, pos_neg_labels, clauses, images
    score_positive = score_positive.permute(0, 3, 2, 1).unsqueeze(2)
    score_negative = score_negative.permute(0, 3, 2, 1).unsqueeze(2)
    all_predicates_scores = torch.cat((score_negative, score_positive), 2)

    return all_predicates_scores


def eval_predicates_sign(c_score):
    resolution = 2

    clause_sign_list = []
    clause_high_scores = []
    clause_score_full_list = []
    for clause_image_score in c_score:
        data_map = np.zeros(shape=[resolution, resolution])
        for index in range(len(clause_image_score[0][0])):
            x_index = int(clause_image_score[0][0][index] * resolution)
            y_index = int(clause_image_score[1][0][index] * resolution)
            data_map[x_index, y_index] += 1

        pos_low_neg_low_area = data_map[0, 0]
        pos_high_neg_low_area = data_map[0, 1]
        pos_low_neg_high_area = data_map[1, 0]
        pos_high_neg_high_area = data_map[1, 1]

        # TODO: find a better score evaluation function
        clause_score = pos_high_neg_low_area + pos_high_neg_high_area * 0.8
        clause_high_scores.append(clause_score)
        clause_score_full_list.append(
            [pos_low_neg_low_area, pos_high_neg_low_area, pos_low_neg_high_area, pos_high_neg_high_area])

        data_map[0, 0] = 0
        if np.max(data_map) == data_map[0, 1] and data_map[0, 1] > data_map[1, 1]:
            clause_sign_list.append(True)
        else:
            clause_sign_list.append(False)

    return clause_sign_list, clause_high_scores, clause_score_full_list


def check_repeat_conflict(atom1, atom2):
    if atom1.terms[0].name == atom2.terms[0].name and atom1.terms[1].name == atom2.terms[1].name:
        return True
    if atom1.terms[0].name == atom2.terms[1].name and atom1.terms[1].name == atom2.terms[0].name:
        return True
    return False


def is_conflict_bodies(pi_bodies, clause_bodies):
    is_conflict = False
    # check for pi_bodies confliction
    # for i, bs_1 in enumerate(pi_bodies):
    #     for j, bs_2 in enumerate(pi_bodies):
    #         if i == j:
    #             continue
    #         is_conflict = check_conflict_body(bs_1, bs_2)
    #         if is_conflict:
    #             return True

    # check for pi_bodies and clause_bodies confliction
    for i, p_b in enumerate(pi_bodies):
        for j, c_b in enumerate(clause_bodies):
            if p_b == c_b and p_b.pred.name != "in":
                is_conflict = True
            elif p_b.pred.name == c_b.pred.name:
                if p_b.pred.name == "rho":
                    is_conflict = check_repeat_conflict(p_b, c_b)
                elif p_b.pred.name == "phi":
                    is_conflict = check_repeat_conflict(p_b, c_b)
            if is_conflict:
                return True
            # if "at_area" in p_b.pred.name and "at_area" in c_b.pred.name:
            #     if p_b.terms == c_b.terms:
            #         return True
            #     elif conflict_pred(p_b.pred.name,
            #                        c_b.pred.name,
            #                        list(p_b.terms),
            #                        list(c_b.terms)):
            #         return True

    return False


def check_conflict_body(b1, b2):
    if "phi" in b1.pred.name and "phi" in b2.pred.name:
        if list(b1.terms) == list(b2.terms):
            return True
        elif conflict_pred(b1.pred.name,
                           b2.pred.name,
                           list(b1.terms),
                           list(b2.terms)):
            return True
    return False


def get_inv_body_preds(inv_body):
    preds = []
    for atom_list in inv_body:
        for atom in atom_list:
            preds.append(atom.pred.name)
            for t in atom.terms:
                if "O" not in t.name:
                    preds.append(t.name)
    return preds


def remove_duplicate_predicates(new_predicates, args):
    non_duplicate_pred = []
    for a_i, [p_a, a_score] in enumerate(new_predicates):
        is_duplicate = False
        for b_i, [p_b, b_score] in enumerate(new_predicates[a_i + 1:]):
            if p_a.name == p_b.name:
                continue
            p_a.body.sort()
            p_b.body.sort()
            p_a_body_preds = get_inv_body_preds(p_a.body)
            p_b_body_preds = get_inv_body_preds(p_b.body)
            if p_a_body_preds == p_b_body_preds:
                is_duplicate = True
        if not is_duplicate:
            non_duplicate_pred.append([p_a, a_score])
        # else:
        #     log_utils.add_lines(f"(remove duplicate predicate) {p_a} {a_score}", args.log_file)
    return non_duplicate_pred


def remove_unaligned_predicates(new_predicates):
    non_duplicate_pred = []
    for a_i, [p_a, p_score] in enumerate(new_predicates):
        b_lens = [len(b) - len(p_a.body[0]) for b in p_a.body]
        if sum(b_lens) == 0:
            non_duplicate_pred.append([p_a, p_score])
    return non_duplicate_pred


def remove_extended_clauses(clauses, p_score):
    clauses = list(clauses)
    non_duplicate_pred = []
    long_clauses_indices = []
    for a_i, c_a in enumerate(clauses):
        for b_i, c_b in enumerate(clauses):
            if a_i == b_i:
                continue
            if set(c_a.body) <= set(c_b.body):
                if b_i not in long_clauses_indices:
                    long_clauses_indices.append(b_i)

    short_clauses_indices = list(set([i for i in range(len(clauses))]) - set(long_clauses_indices))
    clauses = [clauses[i] for i in short_clauses_indices]
    # p_score = [p_score[i] for i in short_clauses_indices]

    return set(clauses)


def get_terms_from_atom(atom):
    terms = []
    for t in atom.terms:
        if t.name not in terms and "O" in t.name:
            terms.append(t.name)
    return terms


def check_accuracy(clause_scores_full, pair_num):
    accuracy = clause_scores_full[:, 1] / pair_num

    return accuracy


def get_best_clauses(clauses, clause_scores, step, args, max_clause):
    target_has_been_found = False
    higher = False
    # clause_accuracy = check_accuracy(clause_scores, total_score)
    sn_scores = clause_scores[config.score_type_index["sn"]].to("cpu")
    if sn_scores.max() == 1.0:
        log_utils.add_lines(f"(BS Step {step}) max clause accuracy: {sn_scores.max()}", args.log_file)
        target_has_been_found = True
        c_indices = [np.argmax(sn_scores)]
        for c_i in c_indices:
            log_utils.add_lines(f"{clauses[c_i]}", args.log_file)
    else:
        new_max_score = sn_scores.max()
        c_indices = [np.argmax(sn_scores)]
        # if len(clauses) != len(clause_scores):
        #     print("break")
        max_scoring_clauses = [[clauses[c_i], clause_scores[:, c_i]] for c_i in c_indices]
        new_max_clause = [new_max_score, max_scoring_clauses]

        if new_max_clause[0] > max_clause[0] and str(new_max_clause[1]) != str(max_clause[1]):
            max_clause = new_max_clause
            higher = True
            log_utils.add_lines(f"(BS Step {step}) (global) max clause accuracy: {sn_scores.max()}",
                                args.log_file)
            for c_i in c_indices:
                log_utils.add_lines(f"{clauses[c_i]}, {clause_scores[:, c_i]}", args.log_file)

        else:
            max_clause = [0.0, []]
            log_utils.add_lines(f"(BS Step {step}) (local) max clause accuracy: {sn_scores.max()}", args.log_file)
            for c_i in c_indices:
                log_utils.add_lines(f"{clauses[c_i]}, {clause_scores[:, c_i]}", args.log_file)

    # clause_dict["nc"] = sorted_clauses(clause_dict["nc"], "nc", args, args.nc_top)
    # clause_dict["sc"] = sorted_clauses(clause_dict["sc"], "sc", args, args.sc_top)
    # clause_dict["nc_good"] = sorted_clauses(clause_dict["nc_good"], "nc_good", args, args.nc_good_top)
    # clause_dict["sc_good"] = sorted_clauses(clause_dict["sc_good"], "sc_good", args, args.sc_good_top)
    # clause_dict["uc"] = sorted_clauses(clause_dict["uc"], "uc", args, args.uc_top)
    # clause_dict["uc_good"] = sorted_clauses(clause_dict["uc_good"], "uc_good", args, args.uc_good_top)

    # log_utils.add_lines(
    #     f"(BS Step {step}) "
    #     f"sn_c: {len(clause_dict['sn'])}, "
    #     f"sn_c_good: {len(clause_dict['sn_good'])}, "
    #     f"n_c: {len(clause_dict['nc'])}, "
    #     f"s_c: {len(clause_dict['sc'])}, "
    #     f"n_c_good: {len(clause_dict['nc_good'])}, "
    #     f"s_c_good: {len(clause_dict['sc_good'])}, "
    #     f"u_c_good: {len(clause_dict['uc_good'])}, "
    #     f"u_c: {len(clause_dict['uc'])}, "
    #     f"conflict: {len(clause_dict['conflict'])}.", args.log_file)

    return max_clause, higher


def sorted_clauses(clause_with_scores, args):
    if len(clause_with_scores) > 0:
        c_sorted = sorted(clause_with_scores, key=lambda x: x[1][2], reverse=True)
        log_utils.add_lines(f"clause number: {len(c_sorted)}", args.log_file)
        # for c in c_sorted:
        #     log_utils.add_lines(f"clause: {c[0]} {c[1]}", args.log_file)
        return c_sorted
    else:
        return []


def extract_clauses_from_bs_clauses(bs_clauses, c_type, args):
    clauses = []
    if len(bs_clauses) == 0:
        return clauses

    for bs_clause in bs_clauses:
        clauses.append(bs_clause[0])
        log_utils.add_lines(f"({c_type}): {bs_clause[0]} {bs_clause[1].reshape(-1)}", args.log_file)

    return clauses


def extract_clauses_from_max_clause(bs_clauses, args):
    clauses = []
    if len(bs_clauses) == 0:
        return clauses

    for bs_clause in bs_clauses:
        clauses.append(bs_clause[0])
        log_utils.add_lines(f"add max clause: {bs_clause[0]}", args.log_file)
    return clauses


def select_top_x_clauses(clause_candidates, c_type, args, threshold=None):
    top_clauses_with_scores = []
    clause_candidates_with_scores_sorted = []
    if threshold is None:
        top_clauses_with_scores = clause_candidates
    else:
        clause_candidates_with_scores = []
        for c_i, c in enumerate(clause_candidates):
            four_scores = get_four_scores(clause_candidates[c_i][1].unsqueeze(0))
            clause_candidates_with_scores.append([c, four_scores])
        clause_candidates_with_scores_sorted = sorted(clause_candidates_with_scores, key=lambda x: x[1][0][1],
                                                      reverse=True)
        clause_candidates_with_scores_sorted = clause_candidates_with_scores_sorted[:threshold]
        for c in clause_candidates_with_scores_sorted:
            top_clauses_with_scores.append(c[0])
    for t_i, t in enumerate(top_clauses_with_scores):
        log_utils.add_lines(f'TOP {(c_type)} {t[0]}, {clause_candidates_with_scores_sorted[t_i][1]}', args.log_file)

    return top_clauses_with_scores


def remove_trivial_clauses(refs_non_conflict, args):
    non_trivial_clauses = []
    for ref in refs_non_conflict:
        preds = get_pred_names_from_clauses(ref)

        if not is_trivial_preds(preds):
            non_trivial_clauses.append(ref)
        # else:
        #     log_utils.add_lines(f"(trivial clause) {ref}", args.log_file)
    return non_trivial_clauses


def get_pred_names_from_clauses(clause, exclude_objects=False):
    preds = []
    for atom in clause.body:
        pred = atom.pred.name
        if "in" == pred:
            continue
        if exclude_objects:
            terms = [t.name for t in atom.terms if "O" not in t.name]
        else:
            terms = [t.name for t in atom.terms]
        if pred not in preds:
            preds.append([pred, terms])
    return preds


def is_trivial_preds(preds_terms):
    term_0 = preds_terms[0][1]
    for [pred, terms] in preds_terms:
        if terms != term_0:
            return False
    preds = [pt[0] for pt in preds_terms]

    # for trivial_set in config.trivial_preds_dict:
    #     is_trivial = True
    #     for pred in trivial_set:
    #         if pred not in preds:
    #             is_trivial = False
    #     if is_trivial:
    #         return True

    return False


def remove_3_zone_only_predicates(new_predicates, args):
    passed_predicates = []
    if len(new_predicates) > 0:
        if len(new_predicates[0]) == 0:
            return []
    for predicate in new_predicates:
        if torch.sum(predicate[1][:3]) > 0:
            passed_predicates.append(predicate)
        # else:
        #     log_utils.add_lines(f"(remove 3 zone only predicate) {predicate[0]} {predicate[1]}", args.log_file)
    return passed_predicates


def keep_1_zone_max_predicates(new_predicates):
    passed_predicates = []
    for predicate in new_predicates:
        if torch.max(predicate[1]) == predicate[1][1]:
            passed_predicates.append(predicate)
    return passed_predicates


def remove_same_four_score_predicates(new_predicates, args):
    passed_predicates = []
    passed_scores = []
    for predicate in new_predicates:
        if predicate[1].tolist() not in passed_scores:
            passed_scores.append(predicate[1].tolist())
            passed_predicates.append(predicate)
        # else:
        #     log_utils.add_lines(f"(remove same four score predicate) {predicate[0]} {predicate[1]}", args.log_file)
    return passed_predicates


def get_clause_unused_args(clause):
    used_args = []
    unused_args = []
    all_args = []
    for body in clause.body:
        if body.pred.name == "in":
            for term in body.terms:
                if term.name != "X" and term not in all_args:
                    all_args.append(term)
        else:
            for term in body.terms:
                if term.name != "X" and term not in used_args:
                    used_args.append(term)
    for arg in all_args:
        if arg not in used_args:
            unused_args.append(arg)
    return unused_args


def remove_duplicate_clauses(refs_i, unused_args, used_args, args):
    non_duplicate_c = []
    for clause in refs_i:
        is_duplicate = False
        for body in clause.body:
            if "in" != body.pred.name:
                if len(body.terms) == 2 and "O" not in body.terms[1].name:
                    # predicate with 1 object arg
                    if len(unused_args) > 0:
                        if not (body.terms[0] == unused_args[0] or body.terms[0] in used_args):
                            is_duplicate = True
                            break
                    # predicate with 2 object args
                elif len(body.terms) == 2 and body.terms[0] in unused_args and body.terms[1] in unused_args:
                    if body.terms[0] not in unused_args[:2] and body.terms[1] not in unused_args:
                        is_duplicate = True
                        break
                elif len(body.terms) == 1 and body.terms[0] in unused_args:
                    if body.terms[0] not in unused_args[:1]:
                        is_duplicate = True
                        break

        if not is_duplicate:
            non_duplicate_c.append(clause)
            # log_utils.add_lines(f'(non duplicate clause) {clause}', args.log_file)
        # else:
        # log_utils.add_lines(f'(duplicate clause) {clause}', args.log_file)
    return non_duplicate_c


def replace_inv_to_equiv_preds(inv_pred):
    equiv_preds = []
    for atom_list in inv_pred.body:
        equiv_preds.append(atom_list)
    return equiv_preds


def get_equivalent_clauses(c):
    equivalent_clauses = [c]
    inv_preds = []
    usual_preds = []
    for atom in c.body:
        if "inv_pred" in atom.pred.name:
            inv_preds.append(atom)
        else:
            usual_preds.append(atom)

    if len(inv_preds) == 0:
        return equivalent_clauses
    else:
        for inv_atom in inv_preds:
            inv_pred_equiv_bodies = replace_inv_to_equiv_preds(inv_atom.pred)
            for equiv_inv_body in inv_pred_equiv_bodies:
                equiv_body = sorted(list(set(equiv_inv_body + usual_preds)))
                equiv_c = Clause(head=c.head, body=equiv_body)
                equivalent_clauses.append(equiv_c)

    return equivalent_clauses


def semantic_same_pred_lists(added_pred_list, new_pred_list):
    is_same = True
    for new_pred in new_pred_list:
        for added_pred in added_pred_list:
            if not new_pred == added_pred:
                is_same = False
                break
    if is_same:
        print("break")
    return is_same


def remove_same_semantic_clauses(clauses):
    semantic_diff_clauses = []
    for c in clauses:
        c_equiv_cs = get_equivalent_clauses(c)
        c.equiv_c_preds = []
        for c_equiv in c_equiv_cs:
            c_equiv_cs_preds = get_pred_names_from_clauses(c_equiv, exclude_objects=True)
            if c_equiv_cs_preds not in c.equiv_c_preds:
                c.equiv_c_preds.append(c_equiv_cs_preds)

    for c in clauses:
        is_same = False
        for added_c in semantic_diff_clauses:
            c_preds = get_pred_names_from_clauses(c)
            added_c_preds = get_pred_names_from_clauses(added_c)
            if c_preds == added_c_preds:
                is_same = True
                break
            elif semantic_same_pred_lists(added_c.equiv_c_preds, c.equiv_c_preds):
                is_same = True
                break
        if not is_same:
            semantic_diff_clauses.append(c)

    return semantic_diff_clauses


def top_select(bs_clauses, args):
    # all_c = bs_clauses['sn'] + bs_clauses['nc'] + bs_clauses['sc'] + bs_clauses['nc_good'] + bs_clauses['sc_good'] + \
    #         bs_clauses['uc'] + bs_clauses['uc_good']

    top_clauses = sorted_clauses(bs_clauses, args)
    top_clauses = top_clauses[:args.c_top]
    top_clauses = extract_clauses_from_max_clause(top_clauses, args)
    return top_clauses


def get_semantic_from_c(clause):
    semantic = []
    semantic += get_pred_names_from_clauses(clause)
    return semantic
