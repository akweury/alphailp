import datetime

import torch

import log_utils
import logic_utils
import chart_utils
import eval_utils


def classify_clauses(clauses, four_scores, all_scores, args, search_type):
    sufficient_necessary_clauses = []
    necessary_clauses = []
    sufficient_clauses = []
    unclassified_clauses = []
    sn_good_clauses = []
    sc_good_clauses = []
    nc_good_clauses = []
    uc_good_clauses = []
    conflict_clauses = []
    for c_i, clause in enumerate(clauses):
        data_size = args.data_size
        # if torch.max(last_3, dim=-1)[0] == last_3[0] and last_3[0] > last_3[2]:
        #     good_clauses.append((clause, scores))
        score = four_scores[c_i]
        if eval_utils.is_sn(score, data_size):
            sufficient_necessary_clauses.append((clause, score, all_scores[c_i]))
            # log_utils.add_lines(f'(sn) {clause}, {four_scores[c_i]}', args.log_file)
        elif eval_utils.is_sn_th_good(score, data_size, args.sn_th):
            sn_good_clauses.append((clause, score, all_scores[c_i]))
            # log_utils.add_lines(f'(sn_good) {clause}, {four_scores[c_i]}', args.log_file)
        elif eval_utils.is_conflict(score, data_size, args.conflict_th):
            conflict_clauses.append((clause, score, all_scores[c_i]))
        elif search_type == "nc":
            if eval_utils.is_nc(score, data_size, 1):
                necessary_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_nc_th_good(score, data_size, args.nc_th):
                nc_good_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_sc(score, data_size, 1):
                sufficient_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_sc_th_good(score, data_size, args.sc_th):
                sc_good_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_uc_th_good(score, args.uc_th):
                uc_good_clauses.append((clause, score, all_scores[c_i]))
                # log_utils.add_lines(f"(uc_good) {clause}, {four_scores[c_i]}", args.log_file)
            else:
                unclassified_clauses.append((clause, score, all_scores[c_i]))
                # log_utils.add_lines(f'(uc) {clause}, {four_scores[c_i]}', args.log_file)
        elif search_type == "sc":
            if eval_utils.is_sc(score, data_size, 1):
                sufficient_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_sc_th_good(score, data_size, args.sc_th):
                sc_good_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_nc(score, data_size, 1):
                necessary_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_nc_th_good(score, data_size, args.nc_th):
                nc_good_clauses.append((clause, score, all_scores[c_i]))
            elif eval_utils.is_uc_th_good(score, args.uc_th):
                uc_good_clauses.append((clause, score, all_scores[c_i]))
                # log_utils.add_lines(f"(uc_good) {clause}, {four_scores[c_i]}", args.log_file)
            else:
                unclassified_clauses.append((clause, score, all_scores[c_i]))
                # log_utils.add_lines(f'(uc) {clause}, {four_scores[c_i]}', args.log_file)
    clause_dict = {"sn": sufficient_necessary_clauses,
                   "nc": necessary_clauses,
                   "sc": sufficient_clauses,
                   "uc": unclassified_clauses,
                   "sn_good": sn_good_clauses,
                   "nc_good": nc_good_clauses,
                   'sc_good': sc_good_clauses,
                   'uc_good': uc_good_clauses,
                   "conflict": conflict_clauses}
    log_utils.add_lines(
        f"sn_c: {len(clause_dict['sn'])}, "
        f"sn_c_good: {len(clause_dict['sn_good'])}, "
        f"n_c: {len(clause_dict['nc'])}, "
        f"s_c: {len(clause_dict['sc'])}, "
        f"n_c_good: {len(clause_dict['nc_good'])}, "
        f"s_c_good: {len(clause_dict['sc_good'])}, "
        f"u_c_good: {len(clause_dict['uc_good'])}, "
        f"u_c: {len(clause_dict['uc'])}, "
        f"conflict: {len(clause_dict['conflict'])}.", args.log_file)
    return clause_dict


def eval_clause_sign(p_scores):
    p_clauses_signs = []

    # p_scores axis: batch_size, pred_names, clauses, pos_neg_labels, images
    p_scores[p_scores == 1] = 0.98
    resolution = 2
    ps_discrete = (p_scores * resolution).int()
    four_zone_scores = torch.zeros((p_scores.size(0), 4))
    img_total = p_scores.size(1)

    # low pos, low neg
    four_zone_scores[:, 0] = img_total - ps_discrete.sum(dim=2).count_nonzero(dim=1)

    # high pos, low neg
    four_zone_scores[:, 1] = img_total - (ps_discrete[:, :, 0] - ps_discrete[:, :, 1] + 1).count_nonzero(dim=1)

    # low pos, high neg
    four_zone_scores[:, 2] = img_total - (ps_discrete[:, :, 0] - ps_discrete[:, :, 1] - 1).count_nonzero(dim=1)

    # high pos, high neg
    four_zone_scores[:, 3] = img_total - (ps_discrete.sum(dim=2) - 2).count_nonzero(dim=1)

    clause_score = four_zone_scores[:, 1] + four_zone_scores[:, 3]

    # four_zone_scores[:, 0] = 0

    # clause_sign_list = (four_zone_scores.max(dim=-1)[1] - 1) == 0

    # TODO: find a better score evaluation function
    p_clauses_signs.append([clause_score, four_zone_scores])

    return p_clauses_signs


def eval_clauses(NSFR, args, pred_names, pos_pred, neg_pred):
    loss_i = 0
    train_size = pos_pred.shape[0]
    bz = args.batch_size_train
    V_T_pos = torch.zeros(len(NSFR.clauses), pos_pred.shape[0], len(NSFR.atoms))
    V_T_neg = torch.zeros(len(NSFR.clauses), pos_pred.shape[0], len(NSFR.atoms))
    for i in range(int(train_size / args.batch_size_train)):
        date_now = datetime.datetime.today().date()
        time_now = datetime.datetime.now().strftime("%H_%M_%S")
        print(f"({date_now} {time_now}) eval batch {i + 1}/{int(train_size / args.batch_size_train)}")
        V_T_pos[:, i * bz:(i + 1) * bz, :] = NSFR.clause_eval_quick(pos_pred[i * bz:(i + 1) * bz])
        V_T_neg[:, i * bz:(i + 1) * bz, :] = NSFR.clause_eval_quick(neg_pred[i * bz:(i + 1) * bz])

    score_positive = NSFR.predict(V_T_pos, pred_names, args.device)
    score_negative = NSFR.predict(V_T_neg, pred_names, args.device)

    if score_positive.size(2) > 1:
        score_positive = score_positive.max(dim=2, keepdim=True)[0]
    if score_negative.size(2) > 1:
        score_negative = score_negative.max(dim=2, keepdim=True)[0]
    score_all = torch.cat((score_negative, score_positive), 2)

    p_clause_signs = eval_clause_sign(score_all)
    clause_score_list, clause_scores_full = p_clause_signs[0]

    return score_all, clause_scores_full
