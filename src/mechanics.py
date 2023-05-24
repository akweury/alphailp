# Created by shaji on 18-Apr-23

from pathlib import Path

import numpy as np
import torch

from rtpt import RTPT

import aitk
import config
import data_hide
import eval_clause_infer
import ilp
import log_utils
import nsfr_utils
import perception
import pi
import visual_utils
from fol import bk
from fol.language import Language
from ilp import ilp_predict
from mechanic_utils import *
from nsfr_utils import get_nsfr_model, get_prob


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
    torch.autograd.set_detect_anomaly(True)
    pm_prediction_dict = perception.get_perception_predictions(args)

    # grouping objects to reduce the problem complexity
    group_val_pos, obj_avail_val_pos = detect_obj_groups_single(args, pm_prediction_dict["val_pos"], "val_pos")
    group_val_neg, obj_avail_val_neg = detect_obj_groups_single(args, pm_prediction_dict["val_neg"], "val_neg")
    group_train_pos, obj_avail_train_pos = detect_obj_groups_single(args, pm_prediction_dict["train_pos"], "train_pos")
    group_train_neg, obj_avail_train_neg = detect_obj_groups_single(args, pm_prediction_dict["train_neg"], "train_neg")
    group_test_pos, obj_avail_test_pos = detect_obj_groups_single(args, pm_prediction_dict["test_pos"], "test_pos")
    group_test_neg, obj_avail_test_neg = detect_obj_groups_single(args, pm_prediction_dict["test_neg"], "test_neg")

    obj_groups = {
        'group_val_pos': group_val_pos,
        'group_val_neg': group_val_neg,
        'group_train_pos': group_train_pos,
        'group_train_neg': group_train_neg,
        'group_test_pos': group_test_pos,
        'group_test_neg': group_test_neg,
    }

    obj_avail_res = {
        'obj_avail_val_pos': obj_avail_val_pos,
        'obj_avail_val_neg': obj_avail_val_neg,
        'obj_avail_train_pos': obj_avail_train_pos,
        'obj_avail_train_neg': obj_avail_train_neg,
        'obj_avail_test_pos': obj_avail_test_pos,
        'obj_avail_test_neg': obj_avail_test_neg,
    }

    # load logical representations
    args.lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lang_base_path = config.root / 'data' / 'lang'

    return args, rtpt, pm_prediction_dict, obj_groups, obj_avail_res


def final_evaluation(NSFR, args):
    # validation split
    log_utils.add_lines(f"Predicting on validation data set...", args.log_file)
    acc_val, rec_val, th_val = ilp.ilp_predict(NSFR, args.val_group_pos, args.val_group_neg,
                                               args, th=0.33, split='val')
    # training split
    log_utils.add_lines(f"Predicting on training data set...", args.log_file)
    acc, rec, th = ilp.ilp_predict(NSFR, args.train_group_pos, args.train_group_neg, args, th=th_val, split='train')
    # test split
    log_utils.add_lines(f"Predicting on test data set...", args.log_file)
    acc_test, rec_test, th_test = ilp.ilp_predict(NSFR, args.test_group_pos, args.test_group_neg, args, th=th_val,
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
    ilp.ilp_main(args, lang)

    # run ilp again
    success = ilp.ilp_test(args, lang)
    if success:
        visualization(args, lang)
        # train nsfr
        NSFR = train_nsfr(args, rtpt, lang)

        return NSFR
    else:
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
    neural_preds = file_utils.load_neural_preds(bk.neural_predicate_2)
    args.neural_preds = [neural_pred for neural_pred in neural_preds]
    # args.neural_preds.append(neural_preds)
    args.p_inv_counter = 0


def train_nsfr(args, rtpt, lang):
    VM = aitk.get_vm(args, lang)
    FC = aitk.get_fc(args, lang, VM)
    NSFR = get_nsfr_model(args, lang, FC, train=True)

    optimizer = torch.optim.RMSprop(NSFR.get_params(), lr=args.lr)
    bce = torch.nn.BCELoss()
    loss_list = []
    stopping_threshold = 1e-4
    test_acc_list = np.zeros(shape=(1, args.epochs))
    # prepare perception result
    train_pos = args.train_group_pos
    train_neg = args.train_group_neg
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

            predicted = get_prob(V_T, NSFR, args)
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

            acc_val, rec_val, th_val = ilp_predict(NSFR, val_pos, val_neg, args, th=0.33, split='val')
            # writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            log_utils.add_lines(f"acc_val:{acc_val} ", args.log_file)
            log_utils.add_lines("Predi$\alpha$ILPcting on training data set...", args.log_file)

            acc, rec, th = ilp_predict(NSFR, train_pos, train_neg, args, th=th_val, split='train')
            # writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            log_utils.add_lines(f"acc_train: {acc}", args.log_file)
            log_utils.add_lines(f"Predicting on test data set...", args.log_file)

            acc, rec, th = ilp_predict(NSFR, test_pos, test_neg, args, th=th_val, split='train')
            # writer.add_scalar("metric/test_acc", acc, global_step=epoch)
            log_utils.add_lines(f"acc_test: {acc}", args.log_file)

    return NSFR


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
            group_pred_image = visual_utils.visual_group_predictions(args, data, input_image, colors, thickness)

            # information image
            info_image = visual_utils.visual_info(lang, input_image.shape, font_size=0.4)

            # adding header and footnotes
            input_image = visual_utils.draw_text(input_image, "input")
            visual_images.append(input_image)

            group_pred_image = visual_utils.draw_text(group_pred_image, f"group:{round(scores[0].tolist(), 4)}")
            group_pred_image = visual_utils.draw_text(group_pred_image, f"{lang.clauses[0]}", position="lower_left",
                                                      font_size=0.4)
            visual_images.append(group_pred_image)

            info_image = visual_utils.draw_text(info_image, f"Info:")
            visual_images.append(info_image)

            # final processing
            final_image = visual_utils.hconcat_resize(visual_images)
            final_image_filename = str(
                args.image_output_path / f"{data_name[0].split('/')[-1].split('.data0.json')[0]}.output.png")
            # visual_utils.show_images(final_image, "Visualization")
            visual_utils.save_image(final_image, final_image_filename)
