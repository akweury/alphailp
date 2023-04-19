# Created by shaji on 18-Apr-23

import config
from mechanic_utils import detect_line_groups, detect_circle_groups, eval_single_group, eval_data
from src.mechanic_utils import test_groups_on_one_image


def detect_obj_groups(args, pattern_pos, pattern_neg):
    # line group
    pattern_lines_pos = detect_line_groups(args, pattern_pos[:, :, config.indices_position])
    pattern_lines_neg = detect_line_groups(args, pattern_neg[:, :, config.indices_position])
    pattern_cir_pos = detect_circle_groups(args, pattern_pos[:, :, config.indices_position])
    pattern_cir_neg = detect_circle_groups(args, pattern_neg[:, :, config.indices_position])

    return pattern_lines_pos, pattern_lines_neg, pattern_cir_pos, pattern_cir_neg


def eval_groups(pattern_pos, pattern_neg, clu_result):
    pattern_lines_pos, pattern_lines_neg, pattern_cir_pos, pattern_cir_neg = clu_result

    shape_group_res = eval_single_group(pattern_lines_pos[:, :, config.group_tensor_shapes],
                                        pattern_lines_neg[:, :, config.group_tensor_shapes])
    color_res = eval_single_group(pattern_pos[:, :, config.indices_color], pattern_neg[:, :, config.indices_color])
    shape_res = eval_single_group(pattern_pos[:, :, config.indices_shape], pattern_neg[:, :, config.indices_shape])

    result = {
        'shape_group': shape_group_res,
        'color': color_res,
        'shape': shape_res
    }

    return result


def check_group_result(args, eval_res_val):
    shape_group_done = eval_res_val['shape_group']["result"] > args.group_conf_th
    color_done = eval_res_val['color']["result"] > args.group_conf_th
    shape_done = eval_res_val['shape']["result"] > args.group_conf_th

    is_done = shape_group_done.sum() + color_done.sum() + shape_done.sum() > 0

    return is_done


def test_groups(test_positive, test_negative, groups):
    accuracy = 0
    for i in range(test_positive.shape[0]):
        accuracy += test_groups_on_one_image(test_positive[i:i + 1], groups)
    for i in range(test_negative.shape[0]):
        neg_score = test_groups_on_one_image(test_negative[i:i + 1], groups)
        accuracy += 1 - neg_score
    accuracy = accuracy / (test_positive.shape[0] + test_negative.shape[0])
    print(f"test acc: {accuracy}")
    return accuracy
