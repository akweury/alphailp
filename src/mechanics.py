# Created by shaji on 18-Apr-23
import os
import torch
from rtpt import RTPT
from pathlib import Path

from mechanic_utils import detect_line_groups, detect_circle_groups, eval_single_group
from mechanic_utils import test_groups_on_one_image, merge_groups, get_args
import nsfr_utils
import log_utils
from nsfr_utils import get_data_pos_loader
from perception import get_perception_predictions
import config


def detect_obj_groups(args, pattern_pos, pattern_neg):
    pattern_lines_pos = detect_line_groups(args, pattern_pos[:, :, config.indices_position])
    pattern_cir_pos = detect_circle_groups(args, pattern_pos[:, :, config.indices_position])
    obj_groups_pos = merge_groups(pattern_lines_pos, pattern_cir_pos, top_group=args.maximum_obj_num)

    pattern_lines_neg = detect_line_groups(args, pattern_neg[:, :, config.indices_position])
    pattern_cir_neg = detect_circle_groups(args, pattern_neg[:, :, config.indices_position])
    obj_groups_neg = merge_groups(pattern_lines_neg, pattern_cir_neg, top_group=args.maximum_obj_num)

    return obj_groups_pos, obj_groups_neg


def eval_groups(pattern_pos, pattern_neg, clu_result):
    group_pos, group_neg = clu_result

    shape_group_res = eval_single_group(group_pos[:, :, config.group_tensor_shapes],
                                        group_neg[:, :, config.group_tensor_shapes])
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


def init():
    args = get_args()
    if args.dataset_type == 'kandinsky':
        if args.small_data:
            name = str(Path("small_KP") / f"NeSy-PI_{args.dataset}")
        else:
            name = str(Path("KP") / f"NeSy-PI_{args.dataset}")
    elif args.dataset_type == "hide":
        name = str(Path("HIDE") / f"NeSy-PI_{args.dataset}")
    else:
        if not args.no_xil:
            name = str(Path('CH') / Path(f"/aILP_{args.dataset}"))
        else:
            name = str(Path('CH') / f"aILP-noXIL_{args.dataset}")

    exp_output_path = config.buffer_path / args.dataset
    if not os.path.exists(exp_output_path):
        os.mkdir(exp_output_path)
    log_file = log_utils.create_log_file(exp_output_path)
    print(f"log_file_path:{log_file}")
    args.log_file = log_file
    log_utils.add_lines(f"args: {args}", log_file)

    if args.no_cuda:
        args.device = torch.device('cpu')
    elif len(str(args.device).split(',')) > 1:
        # multi gpu
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cuda:' + str(args.device))
    log_utils.add_lines(f"device: {args.device}", log_file)

    # Create RTPT object
    rtpt = RTPT(name_initials='JS', experiment_name=name, max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()
    torch.set_printoptions(precision=4)

    pm_prediction_dict = get_perception_predictions(args)

    # load logical representations
    args.lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lang_base_path = config.root / 'data' / 'lang'

    return args, rtpt, pm_prediction_dict
