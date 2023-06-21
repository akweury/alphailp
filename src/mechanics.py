# Created by shaji on 18-Apr-23
import os
import torch
from pathlib import Path
import numpy as np
from rtpt import RTPT

from aitk import percept
from aitk import ai_interface
from aitk.percept_group import detect_obj_groups
from aitk.utils.fol import bk
from aitk.utils import nsfr_utils
from aitk.utils import file_utils
from aitk.utils import visual_utils
from aitk.utils import args_utils

import config
from mechanic_utils import *
import semantic as se


def init():
    args = args_utils.get_args(config.data_path)
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
        file_utils.get_image_names(args)

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

    analysis_path = config.buffer_path / args.dataset_type / args.dataset / "analysis"
    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)
        os.mkdir(analysis_path / "train_pos")
        os.mkdir(analysis_path / "train_neg")
        os.mkdir(analysis_path / "test_pos")
        os.mkdir(analysis_path / "test_neg")
        os.mkdir(analysis_path / "val_pos")
        os.mkdir(analysis_path / "val_neg")

    args.analysis_path = analysis_path

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
    torch.autograd.set_detect_anomaly(True)

    file_path = config.buffer_path / "hide" / f"{args.dataset}"
    pm_prediction_dict = percept.get_perception_predictions(args, file_path)

    # grouping objects to reduce the problem complexity
    group_val_pos, obj_avail_val_pos = detect_obj_groups(args, pm_prediction_dict["val_pos"], "val_pos")
    group_val_neg, obj_avail_val_neg = detect_obj_groups(args, pm_prediction_dict["val_neg"], "val_neg")
    group_train_pos, obj_avail_train_pos = detect_obj_groups(args, pm_prediction_dict["train_pos"], "train_pos")
    group_train_neg, obj_avail_train_neg = detect_obj_groups(args, pm_prediction_dict["train_neg"], "train_neg")
    group_test_pos, obj_avail_test_pos = detect_obj_groups(args, pm_prediction_dict["test_pos"], "test_pos")
    group_test_neg, obj_avail_test_neg = detect_obj_groups(args, pm_prediction_dict["test_neg"], "test_neg")

    group_tensors = {
        'group_val_pos': group_val_pos, 'group_val_neg': group_val_neg,
        'group_train_pos': group_train_pos, 'group_train_neg': group_train_neg,
        'group_test_pos': group_test_pos, 'group_test_neg': group_test_neg,
    }

    group_tensors_indices = {
        'obj_avail_val_pos': obj_avail_val_pos, 'obj_avail_val_neg': obj_avail_val_neg,
        'obj_avail_train_pos': obj_avail_train_pos, 'obj_avail_train_neg': obj_avail_train_neg,
        'obj_avail_test_pos': obj_avail_test_pos, 'obj_avail_test_neg': obj_avail_test_neg,
    }

    # load logical representations
    args.lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lang_base_path = config.root / 'data' / 'lang'

    args.index_pos = config.score_example_index["pos"]
    args.index_neg = config.score_example_index["neg"]

    return args, rtpt, pm_prediction_dict, group_tensors, group_tensors_indices


def final_evaluation(NSFR, args):
    # validation split
    log_utils.add_lines(f"Predicting on validation data set...", args.log_file)

    acc_val, rec_val, th_val = se.run_ilp_predict(args, NSFR, 0.33, "val")
    # training split
    log_utils.add_lines(f"Predicting on training data set...", args.log_file)

    acc, rec, th = se.run_ilp_predict(args, NSFR, th_val, "train")
    # test split
    log_utils.add_lines(f"Predicting on test data set...", args.log_file)
    acc_test, rec_test, th_test = se.run_ilp_predict(args, NSFR, th_val, "test")

    log_utils.add_lines(f"training acc: {acc}, threshold: {th}, recall: {rec}", args.log_file)
    log_utils.add_lines(f"val acc: {acc_val}, threshold: {th_val}, recall: {rec_val}", args.log_file)
    log_utils.add_lines(f"test acc: {acc_test}, threshold: {th_test}, recall: {rec_test}", args.log_file)


def train_and_eval(args, rtpt):
    """
        ilp algorithm: call semantic API
    """
    lang = se.init_language(args, config.pi_type['bk'], "group")
    se.run_ilp_train(args, lang, "group")
    success, clauses = se.run_ilp(args, lang, "group")

    if not success and args.with_explain:
        se.run_ilp_train_explain(args, lang, "object")
        success, clauses = se.run_ilp(args, lang, "object")

    if success:
        scores = se.run_ilp_eval(args, lang, clauses)
        visual_utils.visualization(args, lang, scores)
        # train nsfr
        NSFR = train_nsfr(args, rtpt, lang, clauses)
        return NSFR
    else:
        log_utils.add_lines(f"ILP failed.", args.log_file)
        return None


def update_args(args, pm_prediction_dict, obj_groups, obj_avail):
    args.val_pos = pm_prediction_dict["val_pos"].to(args.device)
    args.val_neg = pm_prediction_dict["val_neg"].to(args.device)
    args.test_pos = pm_prediction_dict["test_pos"].to(args.device)
    args.test_neg = pm_prediction_dict["test_neg"].to(args.device)
    args.train_pos = pm_prediction_dict["train_pos"].to(args.device)
    args.train_neg = pm_prediction_dict["train_neg"].to(args.device)

    args.val_group_pos = obj_groups['group_val_pos']
    args.val_group_neg = obj_groups['group_val_neg']
    args.train_group_pos = obj_groups['group_train_pos']
    args.train_group_neg = obj_groups['group_train_neg']
    args.test_group_pos = obj_groups['group_test_pos']
    args.test_group_neg = obj_groups['group_test_neg']

    args.obj_avail_val_pos = obj_avail['obj_avail_val_pos']
    args.obj_avail_val_neg = obj_avail['obj_avail_val_neg']
    args.obj_avail_train_pos = obj_avail['obj_avail_train_pos']
    args.obj_avail_train_neg = obj_avail['obj_avail_train_neg']
    args.obj_avail_test_pos = obj_avail['obj_avail_test_pos']
    args.obj_avail_test_neg = obj_avail['obj_avail_test_neg']

    args.data_size = args.val_pos.shape[0]
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False
    args.d = len(config.group_tensor_index)

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset

    pi_type = config.pi_type['bk']
    neural_preds = file_utils.load_neural_preds(bk.neural_predicate_2, pi_type)

    args.neural_preds = [neural_pred for neural_pred in neural_preds]
    args.p_inv_counter = 0


def train_nsfr(args, rtpt, lang, clauses):
    VM = ai_interface.get_vm(args, lang)
    FC = ai_interface.get_fc(args, lang, VM, args.group_e)
    NSFR = ai_interface.get_nsfr(args, lang, FC, clauses, train=True)

    optimizer = torch.optim.RMSprop(NSFR.get_params(), lr=args.lr)
    bce = torch.nn.BCELoss()
    loss_list = []
    stopping_threshold = 1e-4
    test_acc_list = np.zeros(shape=(1, args.epochs))
    # prepare perception result
    train_pos = torch.tensor(args.train_group_pos)
    train_neg = torch.tensor(args.train_group_neg)
    test_pos = args.test_group_pos
    test_neg = args.test_group_neg
    val_pos = args.val_group_pos
    val_neg = args.val_group_neg
    train_pred = torch.cat((train_pos, train_neg), dim=0)
    train_label = torch.zeros(len(train_pred)).to(args.device)
    train_label[:len(train_pos)] = 1.0

    for epoch in range(args.epochs):

        # infer and predict the target probability
        loss_i = 0
        train_size = train_pred.shape[0]
        bz = args.batch_size_train
        for i in range(int(train_size / args.batch_size_train)):
            x_data = train_pred[i * bz:(i + 1) * bz]
            y_label = train_label[i * bz:(i + 1) * bz]
            V_T = NSFR(x_data).unsqueeze(0)

            predicted = nsfr_utils.get_prob(V_T, NSFR, args)
            predicted = predicted.squeeze(2)
            predicted = predicted.squeeze(0)
            loss = bce(predicted, y_label)
            loss_i += loss.item()
            loss.backward()
            optimizer.step()
        loss_i = loss_i / (i + 1)
        loss_list.append(loss_i)
        rtpt.step(subtitle=f"loss={loss_i:2.2f}")
        # writer.add_scalar("metric/train_loss", loss_i, global_step=epoch)
        log_utils.add_lines(f"(epoch {epoch}/{args.epochs - 1}) loss: {loss_i}", args.log_file)

        if epoch > 5 and loss_list[epoch - 1] - loss_list[epoch] < stopping_threshold:
            break

        if epoch % 20 == 0:
            NSFR.print_program()
            log_utils.add_lines("Predicting on validation data set...", args.log_file)

            acc_val, rec_val, th_val = se.run_ilp_predict(args, NSFR, th=0.33, split='val')
            log_utils.add_lines(f"acc_val:{acc_val} ", args.log_file)
            log_utils.add_lines("Predi$\alpha$ILPcting on training data set...", args.log_file)

            acc, rec, th = se.run_ilp_predict(args, NSFR, th=th_val, split='train')
            log_utils.add_lines(f"acc_train: {acc}", args.log_file)
            log_utils.add_lines(f"Predicting on test data set...", args.log_file)

            acc, rec, th = se.run_ilp_predict(args, NSFR, th=th_val, split='train')
            log_utils.add_lines(f"acc_test: {acc}", args.log_file)

    return NSFR
