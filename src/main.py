# Created by shaji on 21-Mar-23

import os
import time
import torch
import argparse
from pathlib import Path

from rtpt import RTPT
import datetime

import config
import nsfr_utils
import eval_utils
import mechanics
import src.mechanics
from nsfr_utils import get_data_pos_loader
import log_utils
import file_utils
import pi
from perception import get_perception_predictions
from pi import final_evaluation

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--batch-size-train", type=int,
                        default=20, help="Batch size in nsfr train")
    parser.add_argument("--group_e", type=int, default=4,
                        help="The maximum number of object groups in one image")
    parser.add_argument("--e", type=int, default=5,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", default="red-triangle", help="Use kandinsky patterns dataset")
    parser.add_argument("--dataset-type", default="kandinsky",
                        help="kandinsky or clevr")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--no-pi", action="store_true",
                        help="Generate Clause without predicate invention.")
    parser.add_argument("--small-data", action="store_true",
                        help="Use small training data.")
    parser.add_argument("--score_unique", action="store_false",
                        help="prune same score clauses.")
    parser.add_argument("--semantic_unique", action="store_false",
                        help="prune same semantic clauses.")
    parser.add_argument("--no-xil", action="store_true",
                        help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--small_data", action="store_false",
                        help="Use small portion of valuation data.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--min-beam", type=int, default=0,
                        help="The size of the minimum beam.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--cim-step", type=int, default=5,
                        help="The steps of clause infer module.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1,
                        help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=5,
                        help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101,
                        help="The number of epochs.")
    parser.add_argument("--pi_epochs", type=int, default=3,
                        help="The number of epochs for predicate invention.")
    parser.add_argument("--nc_max_step", type=int, default=3,
                        help="The number of max steps for nc searching.")
    parser.add_argument("--max_step", type=int, default=5,
                        help="The number of max steps for clause searching.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--suff_min", type=float, default=0.1,
                        help="The minimum accept threshold for sufficient clauses.")
    parser.add_argument("--sn_th", type=float, default=0.9,
                        help="The accept threshold for sufficient and necessary clauses.")
    parser.add_argument("--nc_th", type=float, default=0.9,
                        help="The accept threshold for necessary clauses.")
    parser.add_argument("--uc_th", type=float, default=0.8,
                        help="The accept threshold for unclassified clauses.")
    parser.add_argument("--sc_th", type=float, default=0.9,
                        help="The accept threshold for sufficient clauses.")
    parser.add_argument("--sn_min_th", type=float, default=0.2,
                        help="The accept sn threshold for sufficient or necessary clauses.")
    parser.add_argument("--similar_th", type=float, default=1e-3,
                        help="The minimum different requirement between any two clauses.")
    parser.add_argument("--semantic_th", type=float, default=0.75,
                        help="The minimum semantic different requirement between any two clauses.")
    parser.add_argument("--conflict_th", type=float, default=0.9,
                        help="The accept threshold for conflict clauses.")
    parser.add_argument("--length_weight", type=float, default=0.05,
                        help="The weight of clause length for clause evaluation.")
    parser.add_argument("--c_top", type=int, default=20,
                        help="The accept number for clauses.")
    parser.add_argument("--uc_good_top", type=int, default=10,
                        help="The accept number for unclassified good clauses.")
    parser.add_argument("--sc_good_top", type=int, default=20,
                        help="The accept number for sufficient good clauses.")
    parser.add_argument("--sc_top", type=int, default=20,
                        help="The accept number for sufficient clauses.")
    parser.add_argument("--nc_top", type=int, default=10,
                        help="The accept number for necessary clauses.")
    parser.add_argument("--nc_good_top", type=int, default=30,
                        help="The accept number for necessary good clauses.")
    parser.add_argument("--pi_top", type=int, default=20,
                        help="The accept number for pi on each classes.")
    parser.add_argument("--max_cluster_size", type=int, default=4,
                        help="The max size of clause cluster.")
    parser.add_argument("--min_cluster_size", type=int, default=2,
                        help="The min size of clause cluster.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    parser.add_argument("--top_data", type=int, default=20,
                        help="The maximum number of training data.")
    parser.add_argument("--with_bk", action="store_true",
                        help="Using background knowledge by PI.")
    parser.add_argument("--error_th", type=float, default=0.01,
                        help="The threshold for MAE of obj group fitting.")
    parser.add_argument("--line_group_min_sz", type=int, default=3,
                        help="The minimum objects allowed to form a line.")
    parser.add_argument("--cir_group_min_sz", type=int, default=4,
                        help="The minimum objects allowed to form a circle.")
    parser.add_argument("--group_conf_th", type=float, default=0.98,
                        help="The threshold of group confidence.")
    args = parser.parse_args()

    args_file = config.data_path / "lang" / args.dataset_type / args.dataset / "args.json"
    file_utils.load_args_from_file(str(args_file), args)

    return args


def init_args(args, pm_prediction_dict):
    args.val_pos = pm_prediction_dict["val_pos"].to(args.device)
    args.val_neg = pm_prediction_dict["val_neg"].to(args.device)
    args.data_size = args.val_pos.shape[0]
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset
    neural_preds = file_utils.load_neural_preds(str(lang_data_path / 'neural_preds.txt'))[1:]
    args.neural_preds = [[neural_pred] for neural_pred in neural_preds]
    args.neural_preds.append(neural_preds)
    args.p_inv_counter = 0


def main(n):
    args = get_args()
    if args.dataset_type == 'kandinsky':
        if args.small_data:
            name = str(Path("small_KP") / f"NeSy-PI_{args.dataset}_{str(n)}")
        else:
            name = str(Path("KP") / f"NeSy-PI_{args.dataset}_{str(n)}")
    elif args.dataset_type == "hide":
        name = str(Path("HIDE") / f"NeSy-PI_{args.dataset}_{str(n)}")
    else:
        if not args.no_xil:
            name = str(Path('CH') / Path(f"/aILP_{args.dataset}_{str(n)}"))
        else:
            name = str(Path('CH') / f"aILP-noXIL_{args.dataset}_{str(n)}")

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

    if args.no_pi:
        args.pi_epochs = 1

    # run_name = 'predict/' + args.dataset
    # writer = SummaryWriter(str(config.root / "runs" / name), purge_step=0)

    # Create RTPT object
    rtpt = RTPT(name_initials='JS', experiment_name=name,
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()
    torch.set_printoptions(precision=4)
    # get torch data loader
    # train_loader, val_loader, test_loader = get_data_loader(args)

    train_pos_loader, val_pos_loader, test_pos_loader = get_data_pos_loader(args)
    train_neg_loader, val_neg_loader, test_neg_loader = nsfr_utils.get_data_neg_loader(args)

    pm_prediction_dict, pattern_dict = get_perception_predictions(args, val_pos_loader, val_neg_loader,
                                                                  train_pos_loader, train_neg_loader,
                                                                  test_pos_loader, test_neg_loader)
    # load logical representations
    lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lark_path = lark_path
    lang_base_path = config.root / 'data' / 'lang'
    args.lang_base_path = lang_base_path

    init_args(args, pm_prediction_dict)
    obj_groups = mechanics.detect_obj_groups(args, pm_prediction_dict["val_pos"], pm_prediction_dict["val_neg"])
    eval_res_val = mechanics.eval_groups(pm_prediction_dict["val_pos"], pm_prediction_dict["val_neg"], obj_groups)
    is_done = mechanics.check_group_result(args, eval_res_val)
    if False and is_done:
        eval_result_test = mechanics.eval_groups(pm_prediction_dict["test_pos"], pm_prediction_dict["test_neg"],
                                                 obj_groups)
        is_done = mechanics.check_group_result(args, eval_result_test)
        log_utils.print_dataset_simple(args, is_done, eval_result_test)
    else:
        # main program
        start = time.time()
        NSFR = pi.train_and_eval(args, pm_prediction_dict, rtpt, exp_output_path)
        end = time.time()

        log_utils.add_lines(f"=============================", args.log_file)
        log_utils.add_lines(f"Experiment time: {((end - start) / 60):.2f} minute(s)", args.log_file)
        log_utils.add_lines(f"=============================", args.log_file)

        final_evaluation(NSFR, pm_prediction_dict, args)


if __name__ == "__main__":
    main(n=0)
