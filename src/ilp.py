# Created by shaji on 21-Apr-23


from ilp_utils import *
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

        NSFR = nsfr_utils.get_nsfr_model(args, lang, FC)
        # evaluate new clauses
        score_all = eval_clause_infer.eval_clause_on_scenes(NSFR, args, eval_pred)
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
    lang.clause_with_scores = clause_with_scores
    args.is_done = is_done
    args.last_refs = refs
    lang.clauses = refs

    check_result(args, clause_with_scores)

    return refs


def ilp_main(args, lang, with_pi=True):
    for neural_pred in args.neural_preds:
        reset_args(args, lang)
        # grouping objects with new categories
        eval_groups(args)
        lang.update_bk(neural_pred, full_bk=False)
        lang.update_mode_declarations(args)
        # run ilp with pi module
        # searching for a proper clause to describe the pattern.
        while args.iteration < args.max_step and not args.is_done:
            # update system

            VM = aitk.get_vm(args, lang)
            FC = aitk.get_fc(args, lang, VM)

            describe_scenes(args, lang, VM, FC)
            if with_pi:
                ilp_pi(args, lang)

            args.iteration += 1

        # save the promising predicates
        keep_best_preds(args, lang)
        if args.found_ns:
            break

    # print all the invented predicates
    log_utils.print_result(args, lang)


def ilp_pi(args, lang):
    new_pred_with_scores, is_done = cluster_invention(args, lang)
    # convert to strings
    new_clauses_str_list, kp_str_list = generate_new_clauses_str_list(new_pred_with_scores)
    # convert clauses from strings to objects
    # pi_languages = logic_utils.get_pi_clauses_objs(self.args, self.lang, new_clauses_str_list, new_predicates)
    # du = DataUtils(lark_path=args.lark_path, lang_base_path=args.lang_base_path, dataset_type=args.dataset_type,
    #                dataset=args.dataset)
    # lang, vars, init_clauses, atoms = nsfr_utils.get_lang(args)
    # if lang.neural_pred is not None:
    #     lang.preds += lang.neural_pred
    # lang.invented_preds = lang.invented_p
    pi_clauses, pi_kp_clauses = gen_pi_clauses(args, lang, new_pred_with_scores, new_clauses_str_list, kp_str_list)

    lang.pi_clauses += extract_pi(lang, pi_clauses, args)
    lang.pi_kp_clauses = extract_kp_pi(lang, pi_kp_clauses, args)

    if len(lang.invented_preds) > 0:
        # add new predicates
        args.no_new_preds = False
        lang.generate_atoms()

    log_utils.add_lines(f"======  Total PI Number: {len(lang.invented_preds)}  ======", args.log_file)
    for p in lang.invented_preds:
        log_utils.add_lines(f"{p}", args.log_file)

    log_utils.add_lines(f"========== Total {len(lang.pi_clauses)} PI Clauses ============= ", args.log_file)
    for c in lang.pi_clauses:
        log_utils.add_lines(f"{c}", args.log_file)


def ilp_prog():
    return None


def ilp_test(args, lang):
    reset_args(args, lang)
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

    log_utils.print_test_result(args, lang)
