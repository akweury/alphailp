import config
import torch

import config, data_utils

ness_index = config.score_type_index["ness"]
suff_index = config.score_type_index["suff"]
sn_index = config.score_type_index["sn"]


def is_sn(score):
    if score[sn_index] == 1:
        return True
    return False


def is_sn_th_good(score, threshold):
    if score[sn_index] > threshold:
        return True
    else:
        return False


def is_nc(score):
    if score[ness_index] == 1:
        return True
    else:
        return False


def is_nc_th_good(score, threshold):
    if score[ness_index] > threshold:
        return True
    else:
        return False


def is_sc(score):
    if score[suff_index] == 1:
        return True
    else:
        return False


def is_sc_th_good(score, threshold):
    if score[suff_index] > threshold:
        return True
    else:
        return False


def eval_data(data):
    sum_first = data_utils.group_func(data)
    eval_first = data_utils.eval_group_diff(sum_first, dim=0)
    sum_second = data_utils.count_func(sum_first)
    eval_second = data_utils.eval_count_diff(sum_second, dim=0)
    eval_res = torch.tensor([eval_first, eval_second])
    return eval_res, sum_first[0], sum_second[0]


def detect_line(data, same_axis, even_axis, min_line_size=3, shift_th=0.01, even_th=0.15):
    vertical_lines = []
    for data_i in range(data.shape[0]):
        vertical_lines_in_image = []
        line_ranges = []
        objs = data[data_i]
        for obj_i in range(data.shape[1]):
            # check duplications
            if data_utils.in_ranges(objs[obj_i][same_axis], line_ranges):
                continue
            min_value = objs[obj_i][same_axis] - shift_th
            max_value = objs[obj_i][same_axis] + shift_th
            passed_indices = (objs[:, same_axis] > min_value) * (objs[:, same_axis] < max_value)
            if (passed_indices).sum() >= min_line_size:
                points = objs[passed_indices]
                if data_utils.even_distance(points[:, even_axis]) < even_th:
                    vertical_lines_in_image.append(points)
                    line_ranges.append([min_value, max_value])

        vertical_lines.append(vertical_lines_in_image)
    return vertical_lines


def detect_line_model(data, error_th):
    line_groups = []
    group_min_size = 3  # two points (non-collinear) always make a line, Four points make an intended line
    for data_i in range(data.shape[0]):
        image_line_groups = []
        exist_combs = []
        group_indices = data_utils.get_comb(torch.tensor(range(data[data_i].shape[0])), 2).tolist()
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                point_groups, point_indices = data_utils.extend_line_group(group_index, data[data_i], error_th)
                if point_groups is not None and point_groups.shape[0] >= group_min_size:
                    image_line_groups.append(point_groups)
                    exist_combs += data_utils.get_comb(point_indices, 2).tolist()
        line_groups.append(image_line_groups)
    return line_groups

# def detect_line_model(data, same_axis, even_axis, min_line_size=3, area_th=0.1, even_th=0.15):
#     line_groups = []
#     group_min_size = 2
#     for data_i in range(data.shape[0]):
#         lines_per_image = []
#         exist_combs = []
#         objs = data[data_i]
#         # two element combination
#
#         group_indices = data_utils.get_comb(torch.tensor(range(objs.shape[0])), 2).tolist()
#         for g_i, group_index in enumerate(group_indices):
#             # check duplicate
#             if group_index not in exist_combs:
#                 point_groups, point_indices = data_utils.extend_line_group(group_index, objs, area_th)
#                 if point_groups is not None and point_groups.shape[1] > group_min_size:
#                     lines_per_image.append(point_groups)
#                     exist_combs += data_utils.get_comb(point_indices, 2).tolist()
#         line_groups.append(lines_per_image)
#     return line_groups


def detect_circle_model(data, error_th):
    circle_groups = []
    group_min_size = 4  # Three points (non-collinear) always make a circle, Four points make an intended circle
    for data_i in range(data.shape[0]):
        image_circle_groups = []
        exist_combs = []
        group_indices = data_utils.get_comb(torch.tensor(range(data[data_i].shape[0])), 3).tolist()
        for g_i, group_index in enumerate(group_indices):
            # check duplicate
            if group_index not in exist_combs:
                point_groups, point_indices = data_utils.extend_circle_group(group_index, data[data_i], error_th)
                if point_groups is not None and point_groups.shape[0] >= group_min_size:
                    image_circle_groups.append(point_groups)
                    exist_combs += data_utils.get_comb(point_indices, 3).tolist()
        circle_groups.append(image_circle_groups)
    return circle_groups


def eval_data_pos(data):
    # detect vertical line
    lines_groups = detect_line_model(data, error_th=0.01)

    # detect horizontal line
    circle_groups = detect_circle_model(data, error_th=0.01)

    return


def cluster_objects(pattern_pos, pattern_neg):
    # position group
    eval_position_positive_res, score_position_1, score_position_2 = eval_data_pos(
        pattern_pos[:, :, config.indices_position])
    eval_position_negative_res, _, _ = eval_data_pos(pattern_neg[:, :, config.indices_position])
    eval_position_res = data_utils.eval_score(eval_position_positive_res, eval_position_negative_res)

    # color group
    eval_color_positive_res, score_color_1, score_color_2 = eval_data(pattern_pos[:, :, config.indices_color])
    eval_color_negative_res, _, _ = eval_data(pattern_neg[:, :, config.indices_color])
    eval_color_res = data_utils.eval_score(eval_color_positive_res, eval_color_negative_res)

    # shape group
    eval_shape_positive_res, score_shape_1, score_shape_2 = eval_data(pattern_pos[:, :, config.indices_shape])
    eval_shape_negative_res, _, _ = eval_data(pattern_neg[:, :, config.indices_shape])
    eval_shape_res = data_utils.eval_score(eval_shape_positive_res, eval_shape_negative_res)

    predict_color_1 = {"result": eval_color_res[0], "score": score_color_1}
    predict_color_2 = {"result": eval_color_res[1], "score": score_color_2}
    predict_shape_1 = {"result": eval_color_res[0], "score": score_shape_1}
    predict_shape_2 = {"result": eval_shape_res[1], "score": score_shape_2}
    predict_position_1 = {"result": eval_position_res[0], "score": score_position_1}
    predict_position_2 = {"result": eval_position_res[1], "score": score_position_2}

    res = {
        "color_1": predict_color_1,
        "color_2": predict_color_2,
        "shape_1": predict_shape_1,
        "shape_2": predict_shape_2,
        "position_1": predict_position_1,
        "position_2": predict_position_2
    }
    return res


def check_clu_result(clu_result):
    is_done = False
    for pred, res in clu_result.items():
        if res["result"] > 0.99:
            is_done = True
            break
    return is_done


def eval_test_single(data, val_result):
    # color group
    test_score = 1
    eval_color_positive_res, score_color_1, score_color_2 = eval_data(data[:, :, config.indices_color])
    eval_shape_positive_res, score_shape_1, score_shape_2 = eval_data(data[:, :, config.indices_shape])
    for pred, result in val_result.items():
        if result["result"] > 0.99:
            if pred == "color_1":
                if result["score"] != score_color_1:
                    test_score = 0
            elif pred == "color_2":
                if result["score"] != score_color_2:
                    test_score = 0
            elif pred == "shape_1":
                if result["score"] != score_shape_1:
                    test_score = 0
            elif pred == "shape_2":
                if result["score"] != score_shape_2:
                    test_score = 0

    return test_score


def eval_test_dataset(test_positive, test_negative, val_result):
    accuracy = 0
    for i in range(test_positive.shape[0]):
        accuracy += eval_test_single(test_positive[i:i + 1], val_result)
    for i in range(test_negative.shape[0]):
        neg_score = eval_test_single(test_negative[i:i + 1], val_result)
        accuracy += 1 - neg_score
    accuracy = accuracy / (test_positive.shape[0] + test_negative.shape[0])
    print(f"test acc: {accuracy}")
    return accuracy
