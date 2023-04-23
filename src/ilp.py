# Created by shaji on 21-Apr-23


import logic_utils
from ilp_utils import *
from src.fol import mode_declaration
import nsfr_utils
import aitk


def describe_scenes(args, lang, VM, FC):
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

        if is_done:
            break

        # clause evaluation
        NSFR = nsfr_utils.get_nsfr_model(args, lang, refs_extended, FC)
        # evaluate new clauses
        score_all = eval_clause_infer.eval_clause_on_scenes(NSFR, args, eval_pred)
        scores = eval_clause_infer.eval_clauses(score_all[:, :, index_pos], score_all[:, :, index_neg], args, step)
        # classify clauses
        clause_with_scores = eval_clause_infer.prune_low_score_clauses(refs_extended, score_all, scores, args)
        # print best clauses that have been found...
        clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)

        new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
        max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)

        if args.pi_top > 0:
            refs, clause_with_scores, is_done = prune_clauses(clause_with_scores, args)
        else:
            refs = logic_utils.top_select(clause_with_scores, args)
        step += 1

        if found_sn or len(refs) == 0:
            is_done = True
            break

    args.is_done = is_done
    args.last_refs = refs

    clauses = check_result(args, bs_clauses, clauses, max_clause)

    return bs_clauses, clauses, args


def invent_predicates(args, clauses, bs_clauses, pi_clause_generator, new_c, new_p, neural_preds):
    args.no_new_preds = True
    p_new_with_score = []
    lang = pi_clause_generator.lang
    atoms = logic_utils.get_atoms(lang)
    if args.no_pi:
        clauses += logic_utils.extract_clauses_from_bs_clauses(bs_clauses, "clause", args)
    elif args.pi_top > 0:
        # invent new predicate and generate pi clauses
        new_c, new_p, p_new_with_score, is_done = pi_clause_generator.invent_predicate(bs_clauses, new_c, args,
                                                                                       neural_preds, new_p)
        new_pred_num = len(pi_clause_generator.lang.invented_preds) - args.invented_pred_num
        args.invented_pred_num = len(pi_clause_generator.lang.invented_preds)
        if new_pred_num > 0:
            # add new predicates
            args.no_new_preds = False
            lang = pi_clause_generator.lang
            atoms = logic_utils.get_atoms(lang)
            for c in new_c:
                if c not in clauses:
                    clauses.append(c)
    return clauses, new_c, new_p, args, atoms, lang, p_new_with_score


def ilp_main(args, lang,neural_pred, with_pi=True):
    lang.update_bk(neural_pred, full_bk=False)
    lang.update_mode_declarations(args)
    VM = aitk.get_vm(args, lang)
    FC = aitk.get_fc(args, lang, VM)
    # ILP
    # searching for a proper clause to describe the pattern.
    while args.iteration < args.max_step and not args.is_done:
        bs_clauses, clauses, args = describe_scenes(args, lang, VM, FC)
        clauses, pi_clauses, pi_p, args, atoms, lang, p_new_scores = invent_predicates(args, clauses, bs_clauses,
                                                                                       pi_clause_generator,
                                                                                       pi_clauses, pi_p,
                                                                                       args.neural_preds[
                                                                                           neural_pred_i])
        p_inv_with_scores += p_new_scores
        atoms = logic_utils.get_atoms(lang)
        clause_generator, pi_clause_generator, FC = get_models(args, lang, init_clauses, pi_clauses, atoms)
        args.iteration += 1

    p_inv_best = sorted(p_inv_with_scores, key=lambda x: x[1][2], reverse=True)
    p_inv_best = p_inv_best[:args.pi_top]
    p_inv_best = logic_utils.extract_clauses_from_bs_clauses(p_inv_best, "best inv clause", args)

    for new_p in p_inv_best:
        if new_p not in invented_preds:
            invented_preds.append(new_p)
    for new_c in pi_clauses:
        if new_c not in all_pi_clauses and new_c.head.pred in p_inv_best:
            all_pi_clauses.append(new_c)


def ilp_pi():
    return


def ilp_prog():
    return None


def ilp_test(args, lang):
    lang.update_bk(full_bk=True, neural_pred=args.neural_preds)
    lang.update_mode_declarations(args)
    VM = aitk.get_vm(args, lang)
    FC = aitk.get_fc(args, lang, VM)

    # ILP
    # searching for a proper clause to describe the pattern.
    for i in range(args.max_step):
        describe_scenes(args, lang, VM, FC)
        if args.is_done:
            break

    log_utils.print_result(args, lang)
