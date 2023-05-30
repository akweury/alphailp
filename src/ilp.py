# Created by shaji on 21-Apr-23
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, recall_score
from tqdm import tqdm
import torch

from aitk.utils import nsfr_utils

from ilp_utils import *
from pi_utils import gen_clu_pi_clauses, gen_exp_pi_clauses


def describe_scenes(args, lang, FC):
    # generate clauses # time-consuming code
    log_utils.add_lines(f"\n=== beam search iteration {args.iteration}/{args.max_step} ===", args.log_file)
    index_pos = config.score_example_index["pos"]
    index_neg = config.score_example_index["neg"]
    eval_pred = ['kp']
    # extend clauses
    is_done = False
    # if args.no_new_preds:
    step = args.iteration
    refs = args.last_refs
    clause_with_scores = []

    if args.pi_top == 0:
        step = args.iteration
        if len(args.last_refs) > 0:
            refs = args.last_refs

    # describe scenes steps by steps
    while step <= args.iteration:
        # log
        log_utils.print_time(args, args.iteration, step, args.iteration)
        # clause extension
        refs_extended = extend_clauses(args, lang)
        if args.is_done:
            break

        # clause evaluation

        NSFR = ai_interface.get_nsfr(args, lang, FC)
        # evaluate new clauses
        score_all = eval_clause_infer.get_clause_score(NSFR, args, eval_pred)
        scores = eval_clause_infer.eval_clauses(score_all[:, :, index_pos], score_all[:, :, index_neg], args, step)
        # classify clauses
        clause_with_scores = eval_clause_infer.prune_low_score_clauses(refs_extended, score_all, scores, args)
        # print best clauses that have been found...
        clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)

        # new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
        # max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)

        if args.pi_top > 0:
            refs, clause_with_scores, is_done = prune_clauses(clause_with_scores, args)
        else:
            refs = logic_utils.top_select(clause_with_scores, args)
        step += 1
        lang.all_clauses += clause_with_scores
    if len(refs) > 0:
        lang.clause_with_scores = clause_with_scores
        args.is_done = is_done
        args.last_refs = refs

    lang.clauses = args.last_refs

    check_result(args, clause_with_scores)

    return refs


def explain_scenes(args, lang):
    """ explaination should improve the sufficient percentage """
    new_explain_pred_with_scores = explain_invention(args, lang)
    pi_exp_clauses = gen_exp_pi_clauses(args, lang, new_explain_pred_with_scores)
    lang.pi_clauses += pi_exp_clauses


def ilp_extend_cluster(args, lang, neural_pred, level):
    # reset args
    reset_args(args, lang)
    # update predicates
    lang.update_bk(args, neural_pred, full_bk=False)
    # update language
    lang.update_mode_declarations(args)

    # searching for a proper clause to describe the pattern.
    while args.iteration < args.max_step and not args.is_done:

        # update system
        VM = ai_interface.get_vm(args, lang)
        FC = ai_interface.get_fc(args, lang, VM)

        describe_scenes(args, lang, FC)
        # predicate invention by explanation

        if args.with_pi:
            ilp_pi(args, lang)

        args.iteration += 1

    # save the promising predicates
    keep_best_preds(args, lang)


def ilp_clause_explain(args, lang, neural_pred):
    log_utils.add_lines("==================== clause explanation =======================", args.log_file)

    # reset args
    reset_args(args, lang)
    # update predicates
    lang.update_bk(args, neural_pred, full_bk=False)
    # update language
    lang.update_mode_declarations(args)

    # searching for a proper clause to describe the pattern.
    while args.iteration < args.max_step and not args.is_done:

        # update system
        VM = ai_interface.get_vm(args, lang)
        FC = ai_interface.get_fc(args, lang, VM)
        describe_scenes(args, lang, VM, FC)
        # predicate invention by explanation

        if args.with_explain:
            explain_scenes(args, lang)

        if args.with_pi:
            ilp_pi(args, lang)

        args.iteration += 1

    # save the promising predicates
    keep_best_preds(args, lang)


def ilp_main(args, lang, level="group"):
    result = eval_groups(args)

    if level == "group":
        for neural_pred in args.neural_preds:
            ilp_extend_cluster(args, lang, neural_pred, level=level)
            if args.found_ns:
                break

    elif level == "object":
        # for loop: clauses
        for neural_pred in args.neural_preds:
            ilp_clause_explain(args, lang, neural_pred)
            if args.found_ns:
                break
    else:
        raise ValueError

    # print all the invented predicates
    log_utils.print_result(args, lang)
    success = ilp_test(args, lang)

    return success


def ilp_pi(args, lang):
    # predicate invention by clustering
    new_clu_pred_with_scores = cluster_invention(args, lang)
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


def ilp_prog():
    return None


def ilp_test(args, lang):
    log_utils.add_lines(f"================== ILP TEST ==================", args.log_file)
    reset_args(args, lang)
    lang.update_bk(args, full_bk=True, neural_pred=args.neural_preds)
    lang.update_mode_declarations(args)
    VM = ai_interface.get_vm(args, lang)
    FC = ai_interface.get_fc(args, lang, VM)

    # ILP
    # searching for a proper clause to describe the pattern.
    for i in range(args.max_step):
        describe_scenes(args, lang, VM, FC)

        if args.is_done:
            break
    sorted_clauses_with_scores = sorted(lang.all_clauses, key=lambda x: x[1][2], reverse=True)[:args.c_top]
    lang.clauses = [c[0] for c in sorted_clauses_with_scores]

    success = log_utils.print_test_result(args, lang, sorted_clauses_with_scores)
    return success


def ilp_predict(NSFR, pos_pred, neg_pred, args, th=None, split='train'):
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
