# Created by shaji on 18-Apr-23

from pathlib import Path

from rtpt import RTPT

import data_hide
import ilp
import perception
import pi
from fol import bk
from fol.language import Language
from mechanic_utils import *


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

    # get images names
    if args.dataset_type == "hide":
        data_hide.get_image_names(args)

    exp_output_path = config.buffer_path / args.dataset_type / args.dataset / "logs"
    if not os.path.exists(exp_output_path):
        os.mkdir(exp_output_path)
    log_file = log_utils.create_log_file(exp_output_path)
    print(f"log_file_path:{log_file}")
    args.log_file = log_file
    log_utils.add_lines(f"args: {args}", log_file)

    img_output_path = config.buffer_path / args.dataset_type / args.dataset / "image_output"
    if not os.path.exists(img_output_path):
        os.mkdir(img_output_path)
    args.image_output_path = img_output_path

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

    pm_prediction_dict = perception.get_perception_predictions(args)

    # grouping objects to reduce the problem complexity
    obj_groups = detect_obj_groups_with_bk(args, pm_prediction_dict)

    # load logical representations
    args.lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lang_base_path = config.root / 'data' / 'lang'

    return args, rtpt, pm_prediction_dict, obj_groups


def final_evaluation(NSFR, args):
    # validation split
    log_utils.add_lines(f"Predicting on validation data set...", args.log_file)
    acc_val, rec_val, th_val = pi.ilp_predict(NSFR, args.val_group_pos, args.val_group_neg,
                                              args, th=0.33, split='val')
    # training split
    log_utils.add_lines(f"Predicting on training data set...", args.log_file)
    acc, rec, th = pi.ilp_predict(NSFR, args.train_group_pos, args.train_group_neg, args, th=th_val, split='train')
    # test split
    log_utils.add_lines(f"Predicting on test data set...", args.log_file)
    acc_test, rec_test, th_test = pi.ilp_predict(NSFR, args.test_group_pos, args.test_group_neg, args, th=th_val,
                                                 split='test')

    log_utils.add_lines(f"training acc: {acc}, threshold: {th}, recall: {rec}", args.log_file)
    log_utils.add_lines(f"val acc: {acc_val}, threshold: {th_val}, recall: {rec_val}", args.log_file)
    log_utils.add_lines(f"test acc: {acc_test}, threshold: {th_test}, recall: {rec_test}", args.log_file)


def train_and_eval(args, rtpt):
    # init system
    lang = Language(args, [])
    # lang = get_lang_model(args, percept_dict, obj_groups)

    # run ilp
    # ilp.ilp_test(args, lang)

    # invent predicates
    ilp.ilp_main(args, lang, with_pi=True)

    # run ilp again
    ilp.ilp_test(args, lang)
    ilp.visualization(args, lang)
    # train nsfr
    NSFR = pi.train_nsfr(args, rtpt, lang)

    return NSFR


def update_args(args, pm_prediction_dict, obj_groups):
    args.val_pos = pm_prediction_dict["val_pos"].to(args.device)
    args.val_neg = pm_prediction_dict["val_neg"].to(args.device)
    args.test_pos = pm_prediction_dict["test_pos"].to(args.device)
    args.test_neg = pm_prediction_dict["test_neg"].to(args.device)
    args.train_pos = pm_prediction_dict["train_pos"].to(args.device)
    args.train_neg = pm_prediction_dict["train_neg"].to(args.device)

    args.val_group_pos = obj_groups[0]
    args.val_group_neg = obj_groups[1]
    args.train_group_pos = obj_groups[2]
    args.train_group_neg = obj_groups[3]
    args.test_group_pos = obj_groups[4]
    args.test_group_neg = obj_groups[5]

    args.data_size = args.val_pos.shape[0]
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False
    args.d = len(config.group_tensor_index)

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset
    neural_preds = file_utils.load_neural_preds(bk.neural_predicate_2)
    args.neural_preds = [neural_pred for neural_pred in neural_preds]
    # args.neural_preds.append(neural_preds)
    args.p_inv_counter = 0
