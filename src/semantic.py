# Created by jing at 30.05.23

"""
semantic implementation
"""
from aitk.utils.fol.language import Language

import ilp
import converter


def init_language(args, pi_type):
    lang = Language(args, [], pi_type)
    return lang


# def run_ilp(args, lang, level):
#     success = ilp.ilp_main(args, lang, level)
#     return success

def clause2scores():
    pass


def scene2clauses():
    pass


# to be removed
def run_ilp_predict(args, NSFR, th, split):
    acc_val, rec_val, th_val = ilp.ilp_predict(NSFR, args.val_group_pos, args.val_group_neg, args, th=th, split=split)
    return acc_val, rec_val, th_val


# ---------------------------- ilp api --------------------------------------
def extend_clause():
    pass


def reset_args(args):
    ilp.reset_args(args)


def reset_lang(lang, e, neural_pred):
    init_clause = ilp.reset_lang(lang, e, neural_pred)
    return init_clause


def search_clauses(args, lang, init_clauses, FC, level):
    clauses = ilp.search_clauses(args, lang, init_clauses, FC, level)
    return clauses


def explain_clauses(args, lang):
    ilp.explain_scenes(args, lang)


def run_ilp_test(args, lang, level):
    # print all the invented predicates
    success = ilp.ilp_test(args, lang, level)
    return success


def run_ilp_eval(args, lang):
    scores = ilp.ilp_eval(args, lang)
    return scores


def data2tensor_lines(objs):
    line_tensor = converter.to_line_tensor(objs).reshape(-1)

    return line_tensor


def predicate_invention(args, lang, clauses):
    ilp.ilp_pi(args, lang, clauses)


def keep_best_preds(args, lang):
    ilp.keep_best_preds(args, lang)
