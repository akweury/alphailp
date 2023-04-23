# Created by shaji on 21-Apr-23
from logic_utils import get_equivalent_clauses, get_pred_names_from_clauses
import numpy as np
import torch

import config
import log_utils
from refinement import RefinementGenerator

import eval_clause_infer
from fol.logic import *
from fol.data_utils import DataUtils
from infer import InferModule, ClauseInferModule
from eval_clause_infer import eval_clause_sign
from src import log_utils, eval_clause_infer, logic_utils
from src.logic_utils import extract_clauses_from_bs_clauses, top_select, extract_clauses_from_max_clause
from src.mechanic_utils import update_refs
from tensor_encoder import TensorEncoder


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

        if not is_trivial_preds(preds):
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


def check_result(args, bs_clauses, clauses, max_clause):
    if len(bs_clauses) == 0:
        args.is_done = True
    elif len(bs_clauses) > 0 and bs_clauses[0][1][2] == 1.0:
        log_utils.add_lines(f"found sufficient and necessary clause.", args.log_file)
        clauses = extract_clauses_from_bs_clauses([bs_clauses[0]], "sn", args)
        args.is_done = True
        # break
    elif len(bs_clauses) > 0 and bs_clauses[0][1][2] > args.sn_th:
        log_utils.add_lines(f"found quasi-sufficient and necessary clause.", args.log_file)
        for c in bs_clauses:
            if c[1][2] > args.sn_th:
                clauses += extract_clauses_from_bs_clauses([c], "sn_good", args)
        args.is_done = True
        # break
    else:
        if args.pi_top == 0:
            clauses += top_select(bs_clauses, args)
        elif args.iteration == args.max_step:
            clauses += extract_clauses_from_max_clause(bs_clauses, args)
        elif max_clause[1] is not None:
            clauses += extract_clauses_from_max_clause(max_clause[1], args)
        else:
            raise ValueError
    return clauses


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
