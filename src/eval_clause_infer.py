import datetime
import torch

import config


def prune_low_score_clauses(clauses, scores_all, scores, args):
    clause_with_scores = []
    for c_i, clause in enumerate(clauses):
        score = scores[:, c_i]
        if score[0] > args.suff_min:
            clause_with_scores.append((clause, score, scores_all[c_i]))

    return clause_with_scores


# def eval_clause_sign(p_scores):
#     p_clauses_signs = []
#
#     # p_scores axis: batch_size, pred_names, clauses, pos_neg_labels, images
#     p_scores[p_scores == 1] = 0.98
#     resolution = 2
#     ps_discrete = (p_scores * resolution).int()
#     four_zone_scores = torch.zeros((p_scores.size(0), 4))
#     img_total = p_scores.size(1)
#
#     # low pos, low neg
#     four_zone_scores[:, 0] = img_total - ps_discrete.sum(dim=2).count_nonzero(dim=1)
#
#     # high pos, low neg
#     four_zone_scores[:, 1] = img_total - (ps_discrete[:, :, 0] - ps_discrete[:, :, 1] + 1).count_nonzero(dim=1)
#
#     # low pos, high neg
#     four_zone_scores[:, 2] = img_total - (ps_discrete[:, :, 0] - ps_discrete[:, :, 1] - 1).count_nonzero(dim=1)
#
#     # high pos, high neg
#     four_zone_scores[:, 3] = img_total - (ps_discrete.sum(dim=2) - 2).count_nonzero(dim=1)
#
#     clause_score = four_zone_scores[:, 1] + four_zone_scores[:, 3]
#
#     # four_zone_scores[:, 0] = 0
#
#     # clause_sign_list = (four_zone_scores.max(dim=-1)[1] - 1) == 0
#
#     # TODO: find a better score evaluation function
#     p_clauses_signs.append([clause_score, four_zone_scores])
#
#     return p_clauses_signs


def eval_ness(positive_scores):
    positive_scores[positive_scores == 1] = 0.98
    ness_scores = positive_scores.sum(dim=2).sum(dim=1) / positive_scores.shape[1]

    return ness_scores


def eval_suff(negative_scores):
    negative_scores[negative_scores == 1] = 0.98

    # negative scores are inversely proportional to sufficiency scores
    negative_scores_inv = 1 - negative_scores
    suff_scores = negative_scores_inv.sum(dim=2).sum(dim=1) / negative_scores_inv.shape[1]
    return suff_scores


def eval_sn(positive_scores, negative_scores):
    negative_scores[negative_scores == 1] = 0.98
    positive_scores[positive_scores == 1] = 0.98

    # negative scores are inversely proportional to sufficiency scores
    negative_scores_inv = 1 - negative_scores
    ness_scores = positive_scores.sum(dim=2).sum(dim=1) / positive_scores.shape[1]
    suff_scores = negative_scores_inv.sum(dim=2).sum(dim=1) * positive_scores.sum(dim=2).sum(dim=1) / \
                  negative_scores_inv.shape[1]
    return suff_scores


def eval_clauses(score_pos, score_neg, args, c_length):
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
        print(f"({date_now} {time_now}) eval batch {i + 1}/{int(train_size / args.batch_size_train)}")
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


def eval_clause_on_test_scenes(NSFR, args, clause, group_pred, ):
    V_T = NSFR.clause_eval_quick(group_pred)[0, 0]
    preds = [clause.head.pred.name]

    score = NSFR.get_test_target_prediciton(V_T, preds, args.device)
    score[score == 1] = 0.99

    return score


def eval_score_similarity(score, appeared_scores, threshold):
    is_repeat = False
    for appeared_score in appeared_scores:
        if torch.abs(score - appeared_score) / appeared_score < threshold:
            is_repeat = True

    return is_repeat


def eval_semantic_similarity(semantic, appeared_semantics, args):
    is_repeat = False
    for appeared_semantic in appeared_semantics:
        similar_counter = 0
        for p_i in range(len(appeared_semantic)):
            if p_i < len(semantic):
                for a_i in range(len(appeared_semantic[p_i])):
                    if a_i < len(semantic[p_i]):
                        if semantic[p_i][a_i] == appeared_semantic[p_i][a_i]:
                            similar_counter += 1
                        elif isinstance(semantic[p_i][a_i], list):
                            if semantic[p_i][a_i][-1] == appeared_semantic[p_i][a_i][-1]:
                                similar_counter += 1
        similarity = similar_counter / (len(semantic) * 2)
        if similarity > args.semantic_th:
            is_repeat = True
    return is_repeat
