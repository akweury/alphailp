# Created by shaji on 21-Apr-23
from lark import Lark

import config
from mechanic_utils import eval_single_group, check_group_result

from refinement import RefinementGenerator
import log_utils, eval_clause_infer, logic_utils
from fol.language import DataType
from logic_utils import get_equivalent_clauses, get_pred_names_from_clauses
from fol.exp_parser import ExpTree


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
    return non_duplicate_c


def remove_trivial_clauses(refs_non_conflict, args):
    non_trivial_clauses = []
    for ref in refs_non_conflict:
        preds = get_pred_names_from_clauses(ref)

        if not logic_utils.is_trivial_preds(preds):
            non_trivial_clauses.append(ref)
        # else:
        #     log_utils.add_lines(f"(trivial clause) {ref}", args.log_file)
    return non_trivial_clauses


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
                    if "inv_pred" in clause.body[j].pred.name and not is_conflict:
                        pi_name = clause.body[j].pred.name
                        pi_bodies = logic_utils.get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = logic_utils.is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
                    if "inv_pred" in clause.body[i].pred.name and not is_conflict:
                        pi_name = clause.body[i].pred.name
                        pi_bodies = logic_utils.get_pi_bodies_by_name(pi_clauses, pi_name)
                        is_conflict = logic_utils.is_conflict_bodies(pi_bodies, clause.body)
                        if is_conflict:
                            break
                    # if "at_are_6" in clause.body[i].pred.name or "at_are_6" in clause.body[j].pred.name:
                    #     print("conflict")

        if not is_conflict:
            non_conflict_clauses.append(clause)
        # else:
        #     log_utils.add_lines(f"(conflict clause) {clause}", args.log_file)

    return non_conflict_clauses


def extend_clauses(args, lang):
    refs = []
    B_ = []

    refinement_generator = RefinementGenerator(lang=lang)
    for c in lang.clauses:
        refs_i = refinement_generator.refinement_clause(c)
        unused_args, used_args = log_utils.get_unused_args(c)
        refs_i_removed = remove_duplicate_clauses(refs_i, unused_args, used_args, args)
        # remove already appeared refs
        refs_i_removed = list(set(refs_i_removed).difference(set(B_)))
        B_.extend(refs_i_removed)
        refs.extend(refs_i_removed)

    # remove semantic conflict clauses
    refs_no_conflict = remove_conflict_clauses(refs, lang.pi_clauses, args)
    if len(refs_no_conflict) == 0:
        args.is_done = True
    else:
        lang.clauses = refs_no_conflict

    return refs_no_conflict


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


def check_result(args, clauses):
    if len(clauses) == 0:
        args.is_done = True
    elif len(clauses) > 0 and clauses[0][1][2] == 1.0:
        log_utils.add_lines(f"found sufficient and necessary clause.", args.log_file)
        # cs = extract_clauses_from_bs_clauses([clauses[0]], "sn", args)
        args.is_done = True
        # break
    elif len(clauses) > 0 and clauses[0][1][2] > args.sn_th:
        log_utils.add_lines(f"found quasi-sufficient and necessary clause.", args.log_file)
        args.is_done = True
        # for c in clauses:
        # if c[1][2] > args.sn_th:
        #     cs += extract_clauses_from_bs_clauses([c], "sn_good", args)


def prune_clauses(clause_with_scores, args):
    refs = []

    # prune score similar clauses
    log_utils.add_lines(f"=============== score pruning ==========", args.log_file)
    # for c in clause_with_scores:
    #     log_utils.add_lines(f"(clause before pruning) {c[0]} {c[1].reshape(3)}", args.log_file)
    if args.score_unique:
        score_unique_c = []
        score_repeat_c = []
        appeared_scores = []
        for c in clause_with_scores:
            if not eval_clause_infer.eval_score_similarity(c[1][2], appeared_scores, args.similar_th):
                score_unique_c.append(c)
                appeared_scores.append(c[1][2])
            else:
                score_repeat_c.append(c)
        c_score_pruned = score_unique_c
    else:
        c_score_pruned = clause_with_scores

    # prune predicate similar clauses
    log_utils.add_lines(f"=============== semantic pruning ==========", args.log_file)
    if args.semantic_unique:
        semantic_unique_c = []
        semantic_repeat_c = []
        appeared_semantics = []
        for c in c_score_pruned:
            c_semantic = logic_utils.get_semantic_from_c(c[0])
            if not eval_clause_infer.eval_semantic_similarity(c_semantic, appeared_semantics, args):
                semantic_unique_c.append(c)
                appeared_semantics.append(c_semantic)
            else:
                semantic_repeat_c.append(c)
        c_semantic_pruned = semantic_unique_c
    else:
        c_semantic_pruned = c_score_pruned

    c_score_pruned = c_semantic_pruned
    # select top N clauses
    if args.c_top is not None and len(c_score_pruned) > args.c_top:
        c_score_pruned = c_score_pruned[:args.c_top]
    log_utils.add_lines(f"after top select: {len(c_score_pruned)}", args.log_file)

    refs += update_refs(c_score_pruned, args)

    return refs, c_score_pruned, False


def generate_new_predicate(args, lang, clause_clusters, clause_type=None):
    new_predicate = None
    # positive_clauses_exchange = [(c[1], c[0]) for c in positive_clauses]
    # no_hn_ = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0 and c[0][3] == 0]
    # no_hnlp = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0]
    # score clauses properly

    new_predicates = []
    # cluster predicates
    for pi_index, [clause_cluster, cluster_score] in enumerate(clause_clusters):
        p_args = logic_utils.count_arity_from_clause_cluster(clause_cluster)
        dtypes = [DataType("object")] * len(p_args)
        new_predicate = lang.get_new_invented_predicate(args, arity=len(p_args), pi_dtypes=dtypes,
                                                        p_args=p_args, pi_types=clause_type)
        new_predicate.body = []
        for [c_i, clause, c_score] in clause_cluster:
            atoms = []
            for atom in clause.body:
                terms = logic_utils.get_terms_from_atom(atom)
                terms = sorted(terms)
                if "X" in terms:
                    terms.remove("X")
                obsolete_term = [t for t in terms if t not in p_args]
                if len(obsolete_term) == 0:
                    atoms.append(atom)
            new_predicate.body.append(atoms)
        if len(new_predicate.body) > 1:
            new_predicates.append([new_predicate, cluster_score])
        elif len(new_predicate.body) == 1:
            body = (new_predicate.body)[0]
            if len(body) > new_predicate.arity + 1:
                new_predicates.append([new_predicate, cluster_score])
    return new_predicates


def cluster_invention(args, lang):
    found_ns = False

    clu_lists = logic_utils.search_independent_clauses_parallel(args, lang)
    new_predicates = generate_new_predicate(args, lang, clu_lists)
    new_predicates = new_predicates[:args.pi_top]
    lang.invented_preds_with_scores = new_predicates
    return new_predicates, found_ns


def generate_new_clauses_str_list(new_predicates):
    pi_str_lists = []
    kp_str_lists = []
    for [new_predicate, p_score] in new_predicates:
        single_pi_str_list = []
        # head_args = "(O1,O2)" if new_predicate.arity == 2 else "(X)"
        kp_clause = "kp(X):-"
        head_args = "("

        for arg in new_predicate.args:
            head_args += arg + ","
            kp_clause += f"in({arg},X),"
        head_args = head_args[:-1]
        head_args += ")"
        kp_clause += f"{new_predicate.name}{head_args}."
        kp_str_lists.append(kp_clause)

        head = new_predicate.name + head_args + ":-"
        for body in new_predicate.body:
            body_str = ""
            for atom_index in range(len(body)):
                atom_str = str(body[atom_index])
                end_str = "." if atom_index == len(body) - 1 else ","
                body_str += atom_str + end_str
            new_clause = head + body_str
            single_pi_str_list.append(new_clause)
        pi_str_lists.append([single_pi_str_list, p_score])

    return pi_str_lists, kp_str_lists


def extract_pi(lang, all_pi_clauses, args):
    for index, new_p in enumerate(lang.invented_preds):
        if new_p in lang.invented_preds:
            continue
        is_duplicate = False
        for self_p in lang.invented_preds:
            if new_p.body == self_p.body:
                is_duplicate = True
                log_utils.add_lines(f"duplicate pi body {new_p.name} {new_p.body}", args.log_file)
                break
        if not is_duplicate:
            print(f"add new predicate: {new_p.name}")
            lang.invented_preds.append(new_p)
        else:
            log_utils.add_lines(f"duplicate pi: {new_p}", args.log_file)

    new_p_names = [self_p.name for self_p in lang.invented_preds]
    new_all_pi_clausese = []
    for pi_c in all_pi_clauses:
        pi_c_head_name = pi_c.head.pred.name
        if pi_c_head_name in new_p_names:
            new_all_pi_clausese.append(pi_c)
    return new_all_pi_clausese


def extract_kp_pi(new_lang, all_pi_clauses, args):
    new_all_pi_clausese = []
    for pi_c in all_pi_clauses:
        pi_c_head_name = pi_c.head.pred.name
        new_all_pi_clausese.append(pi_c)
    return new_all_pi_clausese


def reset_args(args, lang):
    args.is_done = False
    args.iteration = 0
    args.max_clause = [0.0, None]
    args.no_new_preds = False
    args.last_refs = lang.load_init_clauses(args)

    return args


def update_refs(clause_with_scores, args):
    refs = []
    nc_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_with_scores, "clause", args)
    refs += nc_clauses

    return refs


def eval_groups(args):
    val_pattern_pos = args.val_pos
    val_pattern_neg = args.val_neg
    test_pattern_pos = args.test_pos
    test_pattern_neg = args.test_neg
    group_pos, group_neg = args.group_pos, args.group_neg

    shape_group_res = eval_single_group(group_pos[:, :, config.group_tensor_shapes],
                                        group_neg[:, :, config.group_tensor_shapes])
    color_res = eval_single_group(val_pattern_pos[:, :, config.indices_color],
                                  val_pattern_neg[:, :, config.indices_color])
    shape_res = eval_single_group(val_pattern_pos[:, :, config.indices_shape],
                                  val_pattern_neg[:, :, config.indices_shape])

    result = {
        'shape_group': shape_group_res,
        'color': color_res,
        'shape': shape_res
    }
    is_done = check_group_result(args, result)
    # The pattern is too simple. Print the reason.
    if False and is_done:
        # Dataset is too simple. Finish the program.
        eval_result_test = eval_groups(test_pattern_pos, test_pattern_neg, clu_result)
        is_done = check_group_result(args, eval_result_test)
        log_utils.print_dataset_simple(args, is_done, eval_result_test)

    return result


def gen_pi_clauses(args, lang, new_predicates, clause_str_list_with_score, kp_str_list):
    """Read lines and parse to Atom objects.
    """

    with open(args.lark_path, encoding="utf-8") as grammar:
        lp_clause = Lark(grammar.read(), start="clause")

    for n_p in new_predicates:
        lang.invented_preds.append(n_p[0])
    clause_str_list = []
    for c_str, c_score in clause_str_list_with_score:
        clause_str_list += c_str

    clauses = []
    for clause_str in clause_str_list:
        tree = lp_clause.parse(clause_str)
        clause = ExpTree(lang).transform(tree)
        clauses.append(clause)

    for str in kp_str_list:
        print(str)
    kp_clause = []
    for clause_str in kp_str_list:
        tree = lp_clause.parse(clause_str)
        clause = ExpTree(lang).transform(tree)
        kp_clause.append(clause)

    return clauses, kp_clause
