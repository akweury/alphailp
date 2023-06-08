# Created by shaji on 21-Apr-23
import datetime
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, recall_score
from tqdm import tqdm
import torch

import config
import eval_clause_infer
from aitk import ai_interface
from aitk.utils.fol import bk, logic
from aitk.utils import nsfr_utils
from aitk.utils import lang_utils
from aitk.utils import log_utils
from aitk.utils import eval_utils
from aitk.utils import logic_utils
from aitk.utils.fol.refinement import RefinementGenerator

from ilp_utils import remove_duplicate_clauses, remove_conflict_clauses, update_refs
from pi import generate_explain_pred
from pi_utils import gen_clu_pi_clauses, gen_exp_pi_clauses, generate_new_predicate


def clause_eval(args, lang, FC, clauses, step):
    # clause evaluation
    NSFR = ai_interface.get_nsfr(args, lang, FC, clauses)
    # evaluate new clauses
    eval_predicates = [clauses[0].head.pred.name]
    score_all = get_clause_score(NSFR, args, eval_predicates)
    scores = get_clause_3score(score_all[:, :, args.index_pos], score_all[:, :, args.index_neg], args, step)
    return score_all, scores


def clause_prune(args, clauses, score_all, scores):
    # classify clauses
    clause_with_scores = prune_low_score_clauses(clauses, score_all, scores, args)
    # print best clauses that have been found...
    clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)

    # new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
    # max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)

    if args.pi_top > 0:
        clauses, clause_with_scores = prune_clauses(clause_with_scores, args)
    else:
        clauses = logic_utils.top_select(clause_with_scores, args)

    return clauses, clause_with_scores


def ilp_search(args, lang, init_clauses, FC, level):
    """
    given one or multiple neural predicates, searching for high scoring clauses, which includes following steps
    1. extend given initial clauses
    2. evaluate each clause
    3. prune clauses

    """
    eval_pred = ['kp']
    step = 0
    clause_with_scores = []
    clauses = init_clauses
    while step <= args.iteration:
        # clause extension
        clauses = clause_extend(args, lang, clauses, level)
        if args.is_done:
            break

        # clause evaluation
        score_all, scores = clause_eval(args, lang, FC, clauses, step)
        # classify clauses
        clause_with_scores = prune_low_score_clauses(clauses, score_all, scores, args)
        # print best clauses that have been found...
        clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)

        # new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
        # max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)

        if args.pi_top > 0:
            clauses, clause_with_scores = prune_clauses(clause_with_scores, args)
        else:
            clauses = logic_utils.top_select(clause_with_scores, args)
        step += 1
        lang.all_clauses += clause_with_scores
    if len(clauses) > 0:
        lang.clause_with_scores = clause_with_scores
        # args.last_refs = clauses

    # lang.clauses = args.last_refs
    check_result(args, clause_with_scores)

    return clauses


def explain_scenes(args, lang, clauses):
    """ explaination should improve the sufficient percentage """
    new_explain_pred_with_scores = explain_invention(args, lang, clauses)
    pi_exp_clauses = gen_exp_pi_clauses(args, lang, new_explain_pred_with_scores)
    lang.pi_clauses += pi_exp_clauses


def ilp_pi(args, lang, clauses):
    # predicate invention by clustering
    new_clu_pred_with_scores = cluster_invention(args, lang, clauses)
    # convert to strings
    new_clauses_str_list, kp_str_list = generate_new_clauses_str_list(new_clu_pred_with_scores)
    pi_clu_clauses, pi_kp_clauses = gen_clu_pi_clauses(args, lang, new_clu_pred_with_scores, new_clauses_str_list,
                                                       kp_str_list)
    lang.pi_kp_clauses = extract_kp_pi(lang, pi_kp_clauses, args)
    lang.pi_clauses += pi_clu_clauses

    if len(lang.invented_preds) > 0:
        # add new predicates
        args.no_new_preds = False
        lang.generate_atoms()

    log_utils.add_lines(f"======  Total PI Number: {len(lang.invented_preds)}  ======", args.log_file)
    for p in lang.invented_preds:
        log_utils.add_lines(f"{p}", args.log_file)

    log_utils.add_lines(f"========== Total {len(lang.pi_clauses)} PI Clauses ======== ", args.log_file)
    for c in lang.pi_clauses:
        log_utils.add_lines(f"{c}", args.log_file)


def ilp_test(args, lang, level):
    log_utils.add_lines(f"================== ILP TEST ==================", args.log_file)
    log_utils.print_result(args, lang)

    reset_args(args)
    init_clauses = reset_lang(lang, args.e, args.neural_preds, full_bk=True)

    VM = ai_interface.get_vm(args, lang)
    FC = ai_interface.get_fc(args, lang, VM)
    # ILP
    # searching for a proper clause to describe the pattern.
    for i in range(args.max_step):
        args.iteration = i
        ilp_search(args, lang, init_clauses, FC, level)

        if args.is_done:
            break
    sorted_clauses_with_scores = sorted(lang.all_clauses, key=lambda x: x[1][2], reverse=True)[:args.c_top]

    # lang.clauses = [c[0] for c in sorted_clauses_with_scores]

    success, clauses = log_utils.print_test_result(args, lang, sorted_clauses_with_scores)
    return success, clauses


def ilp_predict(NSFR, args, th=None, split='train'):
    pos_pred = torch.tensor(args.val_group_pos)
    neg_pred = torch.tensor(args.val_group_neg)

    predicted_list = []
    target_list = []
    count = 0

    pm_pred = torch.cat((pos_pred, neg_pred), dim=0)
    train_label = torch.zeros(len(pm_pred))
    train_label[:len(pos_pred)] = 1.0

    target_set = train_label.to(torch.int64)

    for i, sample in tqdm(enumerate(pm_pred, start=0)):
        # to cuda
        sample = sample.unsqueeze(0)
        # infer and predict the target probability
        V_T = NSFR(sample).unsqueeze(0)
        predicted = nsfr_utils.get_prob(V_T, NSFR, args).squeeze(1).squeeze(1)
        predicted_list.append(predicted.detach())
        target_list.append(target_set[i])
        count += V_T.size(0)  # batch size

    predicted_all = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.tensor(target_list).to(torch.int64).detach().cpu().numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted_all, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted_all]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set, [m > thresh for m in predicted_all], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted_all, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted_all])
        rec_score = recall_score(
            target_set, [m > th for m in predicted_all], average=None)
        return accuracy, rec_score, th


def ilp_eval(args, lang, clauses):
    scores_all = []
    target_predicate = [clauses[0].head.pred.name]
    # calculate scores
    for data_type in ["true", "false"]:
        for i in range(len(args.test_group_pos)):
            if data_type == "true":
                data = args.test_group_pos[i]
            else:
                data = args.test_group_neg[i]

            VM = ai_interface.get_vm(args, lang)
            FC = ai_interface.get_fc(args, lang, VM)
            scores_all, scores = clause_eval(args, lang, FC, clauses, 0)
            NSFR = ai_interface.get_nsfr(args, lang, FC, clauses)
            # evaluate new clauses
            scores = eval_utils.eval_clause_on_test_scenes(NSFR, args, lang.clauses[0], data.unsqueeze(0))
            scores_all.append(scores)

    return scores_all


def keep_best_preds(args, lang):
    p_inv_best = sorted(lang.invented_preds_with_scores, key=lambda x: x[1][2], reverse=True)
    p_inv_best = p_inv_best[:args.pi_top]
    p_inv_best = logic_utils.extract_clauses_from_bs_clauses(p_inv_best, "best inv clause", args)

    for new_p in p_inv_best:
        if new_p not in lang.all_invented_preds:
            lang.all_invented_preds.append(new_p)
    for new_c in lang.pi_clauses:
        if new_c not in lang.all_pi_clauses and new_c.head.pred in p_inv_best:
            lang.all_pi_clauses.append(new_c)


def reset_args(args):
    args.is_done = False
    args.iteration = 0
    args.max_clause = [0.0, None]
    args.no_new_preds = False
    args.no_new_preds = True


def reset_lang(lang, e, neural_pred, full_bk):
    lang.all_clauses = []
    lang.invented_preds_with_scores = []
    init_clause = lang.load_init_clauses(e)
    # update predicates
    lang.update_bk(neural_pred, full_bk)
    # update language
    lang.mode_declarations = lang_utils.get_mode_declarations(e, lang)

    return init_clause


def get_clause_3score(score_pos, score_neg, args, c_length=0):
    scores = torch.zeros(size=(3, score_pos.shape[0])).to(args.device)

    # negative scores are inversely proportional to sufficiency scores
    score_negative_inv = 1 - score_neg

    # calculate sufficient, necessary, sufficient and necessary scores
    ness_index = config.score_type_index["ness"]
    suff_index = config.score_type_index["suff"]
    sn_index = config.score_type_index["sn"]
    scores[ness_index, :] = score_pos.sum(dim=1) / score_pos.shape[1]
    scores[suff_index, :] = score_negative_inv.sum(dim=1) / score_negative_inv.shape[1]
    scores[sn_index, :] = scores[0, :] * scores[1, :] + (c_length + 1) * args.length_weight
    return scores


def get_clause_score(NSFR, args, pred_names, pos_group_pred=None, neg_group_pred=None, batch_size=None):
    """ input: clause, output: score """

    if pos_group_pred is None:
        pos_group_pred = args.val_group_pos
    if neg_group_pred is None:
        neg_group_pred = args.val_group_neg
    if batch_size is None:
        batch_size = args.batch_size_train

    train_size = len(pos_group_pred)
    bz = args.batch_size_train
    V_T_pos = torch.zeros(len(NSFR.clauses), train_size, len(NSFR.atoms)).to(args.device)
    V_T_neg = torch.zeros(len(NSFR.clauses), train_size, len(NSFR.atoms)).to(args.device)
    score_all = torch.zeros(size=(V_T_pos.shape[0], V_T_pos.shape[1], 2)).to(args.device)
    for i in range(int(train_size / batch_size)):
        date_now = datetime.datetime.today().date()
        time_now = datetime.datetime.now().strftime("%H_%M_%S")
        # print(f"({date_now} {time_now}) eval batch {i + 1}/{int(train_size / args.batch_size_train)}")
        V_T_pos[:, i * bz:(i + 1) * bz, :] = NSFR.clause_eval_quick(pos_group_pred[i * bz:(i + 1) * bz])
        V_T_neg[:, i * bz:(i + 1) * bz, :] = NSFR.clause_eval_quick(neg_group_pred[i * bz:(i + 1) * bz])

    score_positive = NSFR.get_target_prediciton(V_T_pos, pred_names, args.device)
    score_negative = NSFR.get_target_prediciton(V_T_neg, pred_names, args.device)

    score_negative[score_negative == 1] = 0.99
    score_positive[score_positive == 1] = 0.99

    if score_positive.size(2) > 1:
        score_positive = score_positive.max(dim=2, keepdim=True)[0]
    if score_negative.size(2) > 1:
        score_negative = score_negative.max(dim=2, keepdim=True)[0]

    index_pos = config.score_example_index["pos"]
    index_neg = config.score_example_index["neg"]

    score_all[:, :, index_pos] = score_positive[:, :, 0]
    score_all[:, :, index_neg] = score_negative[:, :, 0]

    return score_all


def prune_low_score_clauses(clauses, scores_all, scores, args):
    clause_with_scores = []
    for c_i, clause in enumerate(clauses):
        score = scores[:, c_i]
        if score[0] > args.suff_min:
            clause_with_scores.append((clause, score, scores_all[c_i]))

    return clause_with_scores


def clause_extend(args, lang, clauses, level):
    refs = []
    B_ = []

    refinement_generator = RefinementGenerator(lang=lang)
    for c in clauses:
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
        refs_no_conflict = clauses
        args.is_done = True

    log_utils.add_lines(f"=============== extended clauses =================", args.log_file)
    for ref in refs_no_conflict:
        log_utils.add_lines(f"{ref}", args.log_file)
    return refs_no_conflict


def prune_clauses(clause_with_scores, args):
    refs = []

    # prune score similar clauses
    if args.score_unique:
        log_utils.add_lines(f"- score pruning ...", args.log_file)
        # for c in clause_with_scores:
        #     log_utils.add_lines(f"(clause before pruning) {c[0]} {c[1].reshape(3)}", args.log_file)
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

    if args.semantic_unique:
        log_utils.add_lines(f"- semantic pruning ... ", args.log_file)
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
    # log_utils.add_lines(f"after top select: {len(c_score_pruned)}", args.log_file)

    refs += update_refs(c_score_pruned, args)

    return refs, c_score_pruned


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


def explain_invention(args, lang, clauses):
    log_utils.add_lines("- (explain clause) -", args.log_file)

    index_pos = config.score_example_index["pos"]
    index_neg = config.score_example_index["neg"]
    explained_clause = []
    for clause, scores, score_all in lang.clause_with_scores:
        increased_score = scores - scores
        if scores[0] > args.sc_th:
            for atom in clause.body:
                if atom.pred.pi_type == config.pi_type['bk']:
                    unclear_pred = atom.pred
                    atom_terms = atom.terms
                    if unclear_pred.name in bk.pred_pred_mapping.keys():
                        new_pred = generate_explain_pred(args, lang, atom_terms, unclear_pred)
                        if new_pred is not None:
                            new_atom = logic.Atom(new_pred, atom_terms)
                            clause.body.append(new_atom)
            VM = ai_interface.get_vm(args, lang)
            FC = ai_interface.get_fc(args, lang, VM)
            NSFR = ai_interface.get_nsfr(args, lang, FC)
            score_all_new = get_clause_score(NSFR, args, ["kp"])
            scores_new = get_clause_3score(score_all_new[:, :, index_pos], score_all_new[:, :, index_neg],
                                           args, 1)
            increased_score = scores_new - scores

        explained_clause.append([clause, scores])
        log_utils.add_lines(f"(clause) {clause} {scores}", args.log_file)
        log_utils.add_lines(f"(score increasing): {increased_score}", args.log_file)
    return explained_clause


def cluster_invention(args, lang, clauses):
    found_ns = False

    clu_lists = search_independent_clauses_parallel(args, lang, clauses)
    new_preds_with_scores = generate_new_predicate(args, lang, clu_lists, pi_type=config.pi_type["clu"])
    new_preds_with_scores = new_preds_with_scores[:args.pi_top]
    lang.invented_preds_with_scores += new_preds_with_scores

    log_utils.add_lines(f"new PI: {len(new_preds_with_scores)}", args.log_file)
    for new_c, new_c_score in new_preds_with_scores:
        log_utils.add_lines(f"{new_c} {new_c_score.reshape(3)}", args.log_file)

    args.is_done = found_ns
    return new_preds_with_scores


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


def extract_kp_pi(new_lang, all_pi_clauses, args):
    new_all_pi_clausese = []
    for pi_c in all_pi_clauses:
        pi_c_head_name = pi_c.head.pred.name
        new_all_pi_clausese.append(pi_c)
    return new_all_pi_clausese


def search_independent_clauses_parallel(args, lang, clauses):
    patterns = logic_utils.get_independent_clusters(args, lang, clauses)
    # trivial: contain multiple semantic identity bodies
    patterns = logic_utils.check_trivial_clusters(patterns)

    # TODO: parallel programming
    index_neg = config.score_example_index["neg"]
    index_pos = config.score_example_index["pos"]

    # evaluate each new pattern
    clu_all = []
    for cc_i, pattern in enumerate(patterns):
        score_neg = torch.zeros((1, pattern[0][2].shape[0], len(pattern))).to(args.device)
        score_pos = torch.zeros((1, pattern[0][2].shape[0], len(pattern))).to(args.device)
        # score_max = torch.zeros(size=(score_neg.shape[0], score_neg.shape[1], 2)).to(args.device)

        for f_i, [c_i, c, c_score] in enumerate(pattern):
            score_neg[0, :, f_i] = c_score[:, index_neg]
            score_pos[0, :, f_i] = c_score[:, index_pos]

        score_neg = score_neg.max(dim=2, keepdims=True)[0]
        score_pos = score_pos.max(dim=2, keepdims=True)[0]

        # score_max[:, :, index_pos] = score_pos[:, :, 0]
        # score_max[:, :, index_neg] = score_neg[:, :, 0]

        score_all = get_clause_3score(score_pos, score_neg, args, len(pattern[0][1].body) - args.e)
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


def ilp_train(args, lang, level):
    for neural_pred in args.neural_preds:
        reset_args(args)
        init_clauses = reset_lang(lang, args.e, neural_pred, full_bk=False)
        while args.iteration < args.max_step and not args.is_done:
            # update system
            VM = ai_interface.get_vm(args, lang)
            FC = ai_interface.get_fc(args, lang, VM)
            clauses = ilp_search(args, lang, init_clauses, FC, level)
            if args.with_pi:
                ilp_pi(args, lang, clauses)
            args.iteration += 1
        # save the promising predicates
        keep_best_preds(args, lang)
        if args.found_ns:
            break


def ilp_train_explain(args, lang, level):
    for neural_pred in args.neural_preds:
        reset_args(args)
        init_clause = reset_lang(lang, args.e, neural_pred, full_bk=False)
        while args.iteration < args.max_step and not args.is_done:
            # update system
            VM = ai_interface.get_vm(args, lang)
            FC = ai_interface.get_fc(args, lang, VM)
            clauses = ilp_search(args, lang, init_clause, FC, level)
            if args.with_explain:
                explain_scenes(args, lang, clauses)
            if args.with_pi:
                ilp_pi(args, lang, clauses)
            args.iteration += 1
        # save the promising predicates
        keep_best_preds(args, lang)
        if args.found_ns:
            break
