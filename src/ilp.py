# Created by shaji on 21-Apr-23


from ilp_utils import *
import nsfr_utils
import aitk
import visual_utils
import copy


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
        lang.all_clauses += clause_with_scores
    if len(refs) > 0:
        lang.clause_with_scores = clause_with_scores
        args.is_done = is_done
        args.last_refs = refs

    lang.clauses = args.last_refs

    check_result(args, clause_with_scores)

    return refs


def ilp_main(args, lang, with_pi=True):
    result = eval_groups(args)
    for neural_pred in args.neural_preds:
        reset_args(args, lang)
        # grouping objects with new categories
        # eval_groups(args)
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
    sorted_clauses_with_scores = sorted(lang.all_clauses, key=lambda x: x[1][2], reverse=True)[:args.c_top]
    lang.clauses = [c[0] for c in sorted_clauses_with_scores]

    log_utils.print_test_result(args, sorted_clauses_with_scores)


def visualization(args, lang, colors=None, thickness=None, radius=None):
    if colors is None:
        # Blue color in BGR
        colors = [
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
        ]
    if thickness is None:
        # Line thickness of 2 px
        thickness = 2
    if radius is None:
        radius = 10


    for data_type in ["true", "false"]:
        for i in range(len(args.test_group_pos)):
            data_name = args.image_name_dict['test'][data_type][i]
            if data_type == "true":
                data = args.test_group_pos[i]
            else:
                data = args.test_group_neg[i]

            # calculate scores
            VM = aitk.get_vm(args, lang)
            FC = aitk.get_fc(args, lang, VM)
            NSFR = nsfr_utils.get_nsfr_model(args, lang, FC)

            # evaluate new clauses
            scores = eval_clause_infer.eval_clause_on_test_scenes(NSFR, args, lang.clauses[0], data.unsqueeze(0))


            visual_images = []
            # input image
            file_prefix = str(config.root / ".." / data_name[0]).split(".data0.json")[0]
            image_file = file_prefix + ".image.png"
            input_image = visual_utils.get_cv_image(image_file)

            # group prediction
            group_image = copy.deepcopy(input_image)
            indice_center_on_screen_x = config.group_tensor_index["x_center_screen"]
            indice_center_on_screen_y = config.group_tensor_index["y_center_screen"]
            screen_points = data[:, [indice_center_on_screen_x, indice_center_on_screen_y]][:args.e, :]
            group_pred_image = visual_utils.draw_circles(group_image, screen_points, radius=radius, color=colors,
                                                         thickness=thickness)

            indice_left_screen_x = config.group_tensor_index["screen_left_x"]
            indice_left_screen_y = config.group_tensor_index["screen_left_y"]
            indice_right_screen_x = config.group_tensor_index["screen_right_x"]
            indice_right_screen_y = config.group_tensor_index["screen_right_y"]

            screen_left_points = data[:, [indice_left_screen_x, indice_left_screen_y]][:args.e, :]
            screen_right_points = data[:, [indice_right_screen_x, indice_right_screen_y]][:args.e, :]
            group_pred_image = visual_utils.draw_lines(group_pred_image, screen_left_points, screen_right_points,
                                                       color=colors, thickness=thickness)

            input_image = visual_utils.draw_text(input_image, "input")
            visual_images.append(input_image)

            group_pred_image = visual_utils.draw_text(group_pred_image, f"group:{round(scores[0].tolist(), 4)}")
            group_pred_image = visual_utils.draw_text(group_pred_image, f"{lang.clauses[0]}",
                                                      position="lower_left", font_size=0.4)
            visual_images.append(group_pred_image)

            # final processing
            final_image = visual_utils.hconcat_resize(visual_images)
            final_image_filename = str(
                args.image_output_path / f"{data_name[0].split('/')[-1].split('.data0.json')[0]}.output.png")
            # visual_utils.show_images(final_image, "Visualization")
            visual_utils.save_image(final_image, final_image_filename)
