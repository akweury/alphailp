import datetime
import torch

import config
import eval_clause_infer
import logic_utils
from fol import bk
from logic_utils import has_term, find_minimum_common_values
from fol import logic
import pi_utils

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def explain_with_group_terms_preds(args, lang, atom_terms, unclear_pred):
    focused_obj_props = bk.pred_obj_mapping[unclear_pred.name]
    if focused_obj_props is None:
        return None
    # check if it is a clause with group term
    if not has_term(unclear_pred, 'group'):
        return None

    explaining_predicates = bk.pred_pred_mapping[unclear_pred.name]

    val_pos_obj = args.val_pos
    val_pos_group = args.val_group_pos
    val_pos_avail = args.obj_avail_val_pos
    group_objs_all_data = torch.zeros(
        size=(val_pos_group.shape[0], val_pos_group.shape[1], val_pos_obj.shape[1], val_pos_obj.shape[2]))

    focused_obj_prop_indices = [config.obj_tensor_index[prop] for prop in focused_obj_props]
    focused_obj_values = torch.zeros(
        size=(val_pos_obj.shape[0], val_pos_group.shape[1], len(focused_obj_prop_indices)))
    for image_i, all_objs in enumerate(val_pos_obj):
        for g_i in range(val_pos_avail.shape[1]):
            group_objs = all_objs[val_pos_avail[image_i][g_i]]
            if len(group_objs) == 0:
                continue
            group_objs_all_data[image_i, g_i, :group_objs.shape[0]] = group_objs
            objs_mean_value = group_objs.mean(dim=0)
            focused_obj_values[image_i, g_i] = objs_mean_value[focused_obj_prop_indices]

    # raw data for explanation
    min_value_set = find_minimum_common_values(focused_obj_values)
    new_pred = pi_utils.generate_new_explain_predicate(args, lang, atom_terms, min_value_set, focused_obj_prop_indices)

    # pred to atom
    new_atom = logic.Atom(new_pred, atom_terms)



    # to generate a set of new predicates, we need
    # bk predicates (mapping from )
    # convert min common indices to predicates

    NSFR = nsfr_utils.get_nsfr_model(args, lang, FC)
    # evaluate new clauses
    score_all = eval_clause_infer.eval_clause_on_scenes(NSFR, args, eval_pred)
    scores = eval_clause_infer.eval_clauses(score_all[:, :, index_pos], score_all[:, :, index_neg], args, step)
    # classify clauses
    clause_with_scores = eval_clause_infer.prune_low_score_clauses(refs_extended, score_all, scores, args)
    # print best clauses that have been found...
    clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)

    # explain the unclear predicates by extending with new predicates
