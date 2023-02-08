import numpy as np
import itertools
from infer import InferModule, ClauseInferModule
from tensor_encoder import TensorEncoder
from fol.logic import *
from fol.data_utils import DataUtils
from fol.language import DataType

p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])


def get_lang(lark_path, lang_base_path, dataset_type, dataset):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = du.load_clauses(str(du.base_path / 'clauses.txt'), lang)
    bk_clauses = du.load_clauses(str(du.base_path / 'bk_clauses.txt'), lang)
    pi_clauses = du.load_pi_clauses(str(du.base_path / 'pi_clauses.txt'), lang)

    # clauses += pi_clauses

    bk = du.load_atoms(str(du.base_path / 'bk.txt'), lang)
    atoms = generate_atoms(lang)
    return lang, clauses, bk_clauses, pi_clauses, bk, atoms


def get_atoms(lang):
    atoms = generate_atoms(lang)
    return atoms


# def update_lang(lang, gen_pi_clauses_str_list):
#     # update invented preds
#
#     # clauses = du.load_clauses(str(du.base_path / 'clauses.txt'), lang)
#     # bk_clauses = du.load_clauses(str(du.base_path / 'bk_clauses.txt'), lang)
#     # pi_clauses = du.load_pi_clauses(str(du.base_path / 'pi_clauses.txt'), lang)
#     #
#     # # clauses += pi_clauses
#     #
#     # bk = du.load_atoms(str(du.base_path / 'bk.txt'), lang)
#     # atoms = generate_atoms(lang)
#     new_lang = None
#     return new_lang


def get_pi_clauses_objs(lang, lark_path, lang_base_path, dataset_type, dataset, clauses_str_list):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    pi_clauses = du.gen_pi_clauses(lang, clauses_str_list)
    for pi_c in pi_clauses:
        print(pi_c)
    # bk = du.load_atoms(str(du.base_path / 'bk.txt'), lang)
    # atoms = generate_atoms(lang)
    return pi_clauses


def get_searched_clauses(lark_path, lang_base_path, dataset_type, dataset):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = du.load_clauses(str(du.base_path / dataset / 'beam_searched.txt'), lang)
    return clauses


def _get_lang(lark_path, lang_base_path, dataset_type, dataset):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = du.get_clauses(lang)
    bk = du.get_bk(lang)
    atoms = generate_atoms(lang)
    return lang, clauses, bk, atoms


def build_infer_module(clauses, bk_clauses, pi_clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    if len(bk_clauses) > 0:
        te_bk = TensorEncoder(lang, atoms, bk_clauses, device=device)
        I_bk = te_bk.encode()
    else:
        te_bk = None
        I_bk = None
    if len(pi_clauses) > 0:
        te_pi = TensorEncoder(lang, atoms, pi_clauses, device=device)
        I_pi = te_pi.encode()
    else:
        te_pi = None
        I_pi = None
    ##I_bk = None
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk, I_pi=I_pi)
    return im


def build_clause_infer_module(clauses, bk_clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    if len(bk_clauses) > 0:
        te_bk = TensorEncoder(lang, atoms, bk_clauses, device=device)
        I_bk = te_bk.encode()
    else:
        te_bk = None
        I_bk = None

    im = ClauseInferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk)
    return im


def build_pi_clause_infer_module(clauses, bk_clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    if len(bk_clauses) > 0:
        te_bk = TensorEncoder(lang, atoms, bk_clauses, device=device)
        I_bk = te_bk.encode()
    else:
        te_bk = None
        I_bk = None

    im = ClauseInferModule(I, m=m, infer_step=infer_step, device=device, train=train, I_bk=I_bk)
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
            # check if args and pred correspond are in the same area
            if pred.dtypes[0].name == 'area':
                if pred.name[0] + pred.name[5:] != args[0].name:
                    continue
            if len(args) == 1 or len(set(args)) == len(args):
                pi_atoms.append(Atom(pred, args))
    return spec_atoms + sorted(atoms) + sorted(pi_atoms)


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
    for i, atom in enumerate(atoms):
        if atom.pred.name == pred_str:
            return i
    assert 1, pred_str + ' not found.'


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]


def remove_conflict_clauses(clauses):
    print("check for conflict clauses...")
    non_conflict_clauses = []
    for clause in clauses:
        is_conflict = False
        for i in range(len(clause.body)):
            for j in range(i + 1, len(clause.body)):
                if "at_area" in clause.body[i].pred.name and "at_area" in clause.body[j].pred.name:
                    if clause.body[i].terms == clause.body[j].terms:
                        is_conflict = True
                        print(f'conflict clause: {clause}')
                        break
                    elif conflict_pred(clause.body[i].pred.name,
                                       clause.body[j].pred.name,
                                       list(clause.body[i].terms),
                                       list(clause.body[j].terms)):
                        is_conflict = True
                        print(f'conflict clause: {clause}')
                        break
            if is_conflict:
                break
        if not is_conflict:
            non_conflict_clauses.append(clause)

    print("end for checking.")
    print("========= All non-conflict clauses ==========")
    for each in non_conflict_clauses:
        print(each)
    print("=============================================")

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


def common_body_pi_clauses(pi_clause_i, pi_clause_j):
    i_body = set([atom.pred.name for atom in pi_clause_i.body])
    j_body = set([atom.pred.name for atom in pi_clause_j.body])
    common_body = set.intersection(i_body, j_body)
    return common_body


def search_independent_clauses(clauses):
    independent_clauses_all = []
    for i_index, clause_i in enumerate(clauses):
        independent_clauses = []
        for j_index, clause_j in enumerate(clauses):
            if clause_j == clause_i:
                continue
            common_body = common_body_pi_clauses(clause_i, clause_j)
            if len(common_body) > 1:
                continue
            independent_clauses.append({clause_j: j_index})
        independent_clauses_all.append({clause_i: independent_clauses})
    return independent_clauses_all


def sub_lists(l):
    # initializing empty list
    comb = []

    # Iterating till length of list
    for i in range(len(l) + 1):
        # Generating sub list
        comb += [list(j) for j in itertools.combinations(l, i)]
    # Returning list
    return comb


def search_cluster_candidates(independent_clauses_all, clause_scores_full):
    print("search_cluster_candidates")
    cluster_candidates = []
    for c_index, clause_dict in enumerate(independent_clauses_all):
        clause = list(clause_dict.keys())[0]
        clause_full_score = clause_scores_full[c_index]
        total_score = np.sum(clause_full_score)
        complementary_clauses = [list(ic_dict.keys())[0] for ic_dict in list(clause_dict.values())[0]]
        complementary_clauses_index = [list(ic_dict.values())[0] for ic_dict in list(clause_dict.values())[0]]
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
            cluster_candidates.append(clause_dict)
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
                    clauses_set = [clause_dict[clause][c_i] for c_i in indices]
                    cluster_candidates.append({
                        "clause": clause,
                        "clause_score": sum_pos_clause,
                        "clause_set": clauses_set,
                        "clause_set_score": subset,
                    })
    new_pi_clauses = []
    for cluster in cluster_candidates:
        cluster_ind = [list(c.keys())[0] for c in cluster["clause_set"]]
        new_pi_clauses.append(
            [cluster["clause"]] + cluster_ind
        )
    return new_pi_clauses
