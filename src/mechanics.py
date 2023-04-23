# Created by shaji on 18-Apr-23
import os
from rtpt import RTPT
from pathlib import Path

from mechanic_utils import *
from perception import get_perception_predictions
import config
import log_utils, file_utils
from fol import bk
from pi import ilp_predict, train_nsfr

import ilp
import aitk


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


def final_evaluation(NSFR, pm_prediction_dict, args):
    # validation split
    log_utils.add_lines(f"Predicting on validation data set...", args.log_file)
    acc_val, rec_val, th_val = ilp_predict(NSFR, pm_prediction_dict["val_pos"], pm_prediction_dict["val_neg"],
                                           args, th=0.33, split='val')
    # training split
    log_utils.add_lines(f"Predicting on training data set...", args.log_file)
    acc, rec, th = ilp_predict(NSFR, pm_prediction_dict["train_pos"], pm_prediction_dict["train_neg"],
                               args, th=th_val, split='train')
    # test split
    log_utils.add_lines(f"Predicting on test data set...", args.log_file)
    acc_test, rec_test, th_test = ilp_predict(NSFR, pm_prediction_dict["test_pos"], pm_prediction_dict["test_neg"],
                                              args, th=th_val, split='test')

    log_utils.add_lines(f"training acc: {acc}, threshold: {th}, recall: {rec}", args.log_file)
    log_utils.add_lines(f"val acc: {acc_val}, threshold: {th_val}, recall: {rec_val}", args.log_file)
    log_utils.add_lines(f"test acc: {acc_test}, threshold: {th_test}, recall: {rec_test}", args.log_file)


def train_and_eval(args, percept_dict, obj_groups, rtpt):
    # init system
    lang = Language(args, [])
    # lang = get_lang_model(args, percept_dict, obj_groups)
    args = reset_args(args, lang)
    # run ilp
    ilp.ilp_test(args, lang)

    # invent predicates
    for neural_pred in args.neural_preds:
        # grouping objects with new categories
        group_eval_res = eval_groups(args, percept_dict, obj_groups)

        # run ilp with pi module
        ilp.ilp_main(args, lang,neural_pred, with_pi=True)

        if args.found_ns:
            break

    # run ilp again
    ilp.ilp_test(args, lang)

    # train nsfr
    NSFR = train_nsfr(args, percept_dict, rtpt)

    return NSFR


def update_args(args, pm_prediction_dict, obj_groups):
    args.val_pos = pm_prediction_dict["val_pos"].to(args.device)
    args.val_neg = pm_prediction_dict["val_neg"].to(args.device)
    args.group_pos = obj_groups[0]
    args.group_neg = obj_groups[1]
    args.data_size = args.val_pos.shape[0]
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False
    args.d = len(config.group_tensor_index)

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset
    neural_preds = file_utils.load_neural_preds(bk.neural_predicate_2)
    args.neural_preds = [[neural_pred] for neural_pred in neural_preds]
    args.neural_preds.append(neural_preds)
    args.p_inv_counter = 0
