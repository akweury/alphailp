# Created by jing at 30.05.23

"""
semantic implementation
"""
import aitk.utils.data_utils
from aitk.utils.fol.language import Language

import ilp


def init_language(args, pi_type, level):
    lang = Language(args, [], pi_type, level)
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
    acc_val, rec_val, th_val = ilp.ilp_predict(NSFR, args, th=th, split=split)
    return acc_val, rec_val, th_val


# ---------------------------- ilp api --------------------------------------
def extend_clause():
    pass


def reset_args(args):
    ilp.reset_args(args)


def reset_lang(lang, args, level, neural_pred, full_bk):
    init_clause, e = ilp.reset_lang(lang,args, level, neural_pred, full_bk)
    return init_clause, e


def search_clauses(args, lang, init_clauses, FC, level):
    clauses = ilp.ilp_search(args, lang, init_clauses, FC, level)
    return clauses


def explain_clauses(args, lang, clauses):
    ilp.explain_scenes(args, lang, clauses)


def run_ilp(args, lang, level):
    # print all the invented predicates
    success, clauses = ilp.ilp_test(args, lang, level)
    return success, clauses


def run_ilp_eval(args, lang, clauses):
    scores = ilp.ilp_eval(args, lang, clauses)
    return scores


def data2tensor_lines(objs):
    line_tensor = aitk.utils.data_utils.to_line_tensor(objs).reshape(-1)

    return line_tensor


def predicate_invention(args, lang, clauses, e):
    ilp.ilp_pi(args, lang, clauses, e)


def keep_best_preds(args, lang):
    ilp.keep_best_preds(args, lang)


def run_ilp_train(args, lang, level):
    ilp.ilp_train(args, lang, level)


def run_ilp_train_explain(args, lang, level):
    ilp.ilp_train_explain(args, lang, level)
