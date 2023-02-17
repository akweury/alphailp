import os
import torch
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rtpt import RTPT

import config
import pi_utils
import nsfr_utils
import percept
from nsfr_utils import denormalize_kandinsky, get_data_loader, get_data_pos_loader, get_prob, get_nsfr_model, \
    update_initial_clauses
from nsfr_utils import save_images_with_captions, to_plot_images_kandinsky, generate_captions
import logic_utils
from logic_utils import get_lang, get_searched_clauses
from mode_declaration import get_mode_declarations
from clause_generator import ClauseGenerator, PIClauseGenerator
import facts_converter
from percept import YOLOPerceptionModule
from valuation import YOLOValuationModule, PIValuationModule
import chart_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int,
                        default=1, help="Batch size in beam search")
    parser.add_argument("--batch-size-train", type=int,
                        default=10, help="Batch size in nsfr train")
    parser.add_argument("--e", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", default="red-triangle",
                        choices=["twopairs", "threepairs", "red-triangle", "closeby", "closeby-learn",
                                              "online", "online-pair", "nine-circles", "clevr-hans0", "clevr-hans1",
                                              "clevr-hans2"], help="Use kandinsky patterns dataset")
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
    parser.add_argument("--no-xil", action="store_true",
                        help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4,
                        help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5,
                        help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50,
                        help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1,
                        help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=4,
                        help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=21,
                        help="The number of epochs.")
    parser.add_argument("--pi_epochs", type=int, default=3,
                        help="The number of epochs for predicate invention.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200,
                        help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true",
                        help="Using pre searched clauses.")
    args = parser.parse_args()
    return args


# def get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False):


def discretise_NSFR(NSFR, args, device):
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, clauses_, bk_clauses, pi_clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.dataset_type, args.dataset)
    # Discretise NSFR rules
    clauses = NSFR.get_clauses()
    return get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device, train=False)


def predict(NSFR, pos_pred, neg_pred, args, th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0
    ###NSFR = discretise_NSFR(NSFR, args, device)
    # NSFR.print_program()

    pm_pred = torch.cat((pos_pred, neg_pred), dim=0)
    train_label = torch.zeros(len(pm_pred))
    train_label[:len(pos_pred)] = 1.0
    # TODO: check this segment code.

    test_size = pm_pred.shape[0]
    bz = 1
    predicted_all = torch.zeros(pm_pred.size()[0])
    target_set = train_label.to(torch.int64)

    for i, sample in tqdm(enumerate(pm_pred, start=0)):
        # to cuda
        sample = sample.unsqueeze(0)
        # infer and predict the target probability
        V_T = NSFR(sample).unsqueeze(0)
        predicted = get_prob(V_T, NSFR, args).squeeze(1).squeeze(1)
        predicted_list.append(predicted.detach())
        target_list.append(target_set[i])
        count += V_T.size(0)  # batch size

    predicted_all = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.tensor(target_list).to(torch.int64).detach().cpu().numpy()

    # for i in range(int(test_size / args.batch_size_train)):
    #     x_data = pm_pred[i * bz:(i + 1) * bz]
    #     y_label = train_label[i * bz:(i + 1) * bz]
    #     V_T = NSFR(x_data).unsqueeze(0)
    #     predicted = get_prob(V_T, NSFR, args)
    #     predicted = predicted.squeeze(2)
    #     predicted = predicted.squeeze(0)
    #     predicted = predicted
    #     # train_label = train_label.detach().to(torch.int64).to("cpu").numpy()
    #     count += V_T.size(0)  # batch size
    #     predicted_all[i * bz:(i + 1) * bz] = predicted
    #
    # predicted_all = predicted_all.detach().to("cpu").numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted_all, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted_all]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set, [m > thresh for m in predicted_all], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted_all, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted_all])
        rec_score = recall_score(
            target_set, [m > th for m in predicted_all], average=None)
        return accuracy, rec_score, th


def train_nsfr(args, NSFR, pm_prediction_dict, writer, rtpt, exp_output_path):
    optimizer = torch.optim.RMSprop(NSFR.get_params(), lr=args.lr)
    bce = torch.nn.BCELoss()
    loss_list = []
    stopping_threshold = 1e-6
    test_acc_list = np.zeros(shape=(1, args.epochs))
    # prepare perception result
    train_pred = torch.cat((pm_prediction_dict['train_pos'], pm_prediction_dict['train_neg']), dim=0)
    train_label = torch.zeros(len(train_pred)).to(args.device)
    train_label[:len(pm_prediction_dict['train_pos'])] = 1.0

    for epoch in range(args.epochs):

        # for i, sample in tqdm(enumerate(train_pred, start=0)):
        #     # infer and predict the target probability
        #     sample = sample.unsqueeze(0)
        #     V_T = NSFR(sample)
        #     # watch out for PI values
        #     a = V_T.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        #
        #     # NSFR.print_valuation_batch(V_T)
        #     predicted = get_prob(V_T, NSFR, args)
        #     loss = bce(predicted, train_label[i])
        #     loss_i += loss.item()
        #     loss.backward()
        #     # TODO: problem: performs good in positive but bad in negative
        #     optimizer.step()

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
            # TODO: problem: performs good in positive but bad in negative
            optimizer.step()
        loss_i = loss_i / (i + 1)
        loss_list.append(loss_i)
        rtpt.step(subtitle=f"loss={loss_i:2.2f}")
        writer.add_scalar("metric/train_loss", loss_i, global_step=epoch)
        print(f"(epoch {epoch}/{args.epochs - 1}) loss: {loss_i}")

        if epoch > 5 and loss_list[epoch - 1] - loss_list[epoch] < stopping_threshold:
            break

        # print("Predicting on test data set...")
        # acc, rec, th = predict(NSFR, pm_prediction_dict['test_pos'],
        #                        pm_prediction_dict['test_neg'], args, th=0.33, split='train')
        # test_acc_list[0, epoch] = acc
        # chart_utils.plot_line_chart(test_acc_list, str(exp_output_path), labels="Test_Accuracy",
        #                             title=f"Test Accuracy ({args.dataset})", cla_leg=True)
        # NSFR.print_program()
        if epoch % 20 == 0:
            NSFR.print_program()
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(NSFR, pm_prediction_dict['val_pos'],
                                               pm_prediction_dict['val_neg'], args, th=0.33, split='val')
            writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            print("acc_val: ", acc_val)

            print("Predi$\alpha$ILPcting on training data set...")
            acc, rec, th = predict(NSFR, pm_prediction_dict['train_pos'],
                                   pm_prediction_dict['train_neg'], args, th=th_val, split='train')
            writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            print("acc_train: ", acc)

            print("Predicting on test data set...")
            acc, rec, th = predict(NSFR, pm_prediction_dict['test_pos'],
                                   pm_prediction_dict['test_neg'], args, th=th_val, split='train')
            writer.add_scalar("metric/test_acc", acc, global_step=epoch)
            print("acc_test: ", acc)

    return loss


def train_pi(args, PI, optimizer, train_loader, val_loader, test_loader, device, writer, rtpt):
    bce = torch.nn.BCELoss()
    loss_list = []
    loss = None
    for epoch in range(args.epochs):
        loss_i = 0
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            # to cuda
            imgs, target_set = map(lambda x: x.to(device), sample)
            # infer and predict the target probability
            V_T = PI(imgs)
            # NSFR.print_valuation_batch(V_T)
            predicted = get_prob(V_T, PI, args)
            loss = bce(predicted, target_set)
            loss_i += loss.item()
            loss.backward()

            optimizer.step()

            # if i % 20 == 0:
            #    NSFR.print_valuation_batch(V_T)
            #    print("predicted: ", np.round(predicted.detach().cpu().numpy(), 2))
            #    print("target: ", target_set.detach().cpu().numpy())
            #    NSFR.print_program()
            #    print("loss: ", loss.item())

            # print("Predicting on validation data set...")
            # acc_val, rec_val, th_val = predict(
            #    NSFR, val_loader, args, device, writer, th=0.33, split='val')
            # print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
        loss_list.append(loss_i)
        rtpt.step(subtitle=f"loss={loss_i:2.2f}")
        writer.add_scalar("metric/train_loss", loss_i, global_step=epoch)
        print("loss: ", loss_i)
        # NSFR.print_program()
        if epoch % 20 == 0:
            PI.print_program()
            print("Predicting on validation data set...")
            acc_val, rec_val, th_val = predict(PI, val_loader, args, device, th=0.33, split='val')
            writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            print("acc_val: ", acc_val)

            print("Predicting on training data set...")
            acc, rec, th = predict(PI, train_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            print("acc_train: ", acc)

            print("Predicting on test data set...")
            acc, rec, th = predict(PI, test_loader, args, device, th=th_val, split='train')
            writer.add_scalar("metric/test_acc", acc, global_step=epoch)
            print("acc_test: ", acc)

    return loss


def final_evaluation(NSFR, pm_prediction_dict, args):
    # validation split
    print("Predicting on validation data set...")
    acc_val, rec_val, th_val = predict(NSFR, pm_prediction_dict["val_pos"], pm_prediction_dict["val_neg"],
                                       args, th=0.33, split='val')
    # training split
    print("Predicting on training data set...")
    acc, rec, th = predict(NSFR, pm_prediction_dict["train_pos"], pm_prediction_dict["train_neg"],
                           args, th=th_val, split='train')
    # test split
    print("Predicting on test data set...")
    acc_test, rec_test, th_test = predict(NSFR, pm_prediction_dict["test_pos"], pm_prediction_dict["test_neg"],
                                          args, th=th_val, split='test')

    print("training acc: ", acc, "threashold: ", th, "recall: ", rec)
    print("val acc: ", acc_val, "threashold: ", th_val, "recall: ", rec_val)
    print("test acc: ", acc_test, "threashold: ", th_test, "recall: ", rec_test)


def get_perception_predictions(args, val_pos_loader, val_neg_loader, train_pos_loader, train_neg_loader,
                               test_pos_loader, test_neg_loader):
    pm_val_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_val.pth.tar")
    pm_train_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_train.pth.tar")
    pm_test_res_file = str(config.buffer_path / f"{args.dataset}_pm_res_test.pth.tar")

    val_pos_pred, val_neg_pred = percept.eval_images(args, pm_val_res_file, args.device, val_pos_loader, val_neg_loader)
    train_pos_pred, train_neg_pred = percept.eval_images(args, pm_train_res_file, args.device, train_pos_loader,
                                                         train_neg_loader)
    test_pos_pred, test_neg_pred = percept.eval_images(args, pm_test_res_file, args.device, test_pos_loader,
                                                       test_neg_loader)
    pm_prediction_dict = {
        'val_pos': val_pos_pred,
        'val_neg': val_neg_pred,
        'train_pos': train_pos_pred,
        'train_neg': train_neg_pred,
        'test_pos': test_pos_pred,
        'test_neg': test_neg_pred
    }

    return pm_prediction_dict


def get_models(args, lang, val_pos_loader, val_neg_loader,
               clauses, bk_clauses, pi_clauses, atoms, bk):
    PM = YOLOPerceptionModule(e=args.e, d=11, device=args.device)
    VM = YOLOValuationModule(lang=lang, device=args.device, dataset=args.dataset)
    PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset)
    FC = facts_converter.FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
                                        pi_valuation_module=PI_VM, device=args.device)
    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, pi_clauses, FC)
    PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, bk, bk_clauses, pi_clauses, FC)

    mode_declarations = get_mode_declarations(args, lang, args.n_obj)
    clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, val_pos_loader, val_neg_loader,
                                       mode_declarations,
                                       bk_clauses, no_xil=args.no_xil)  # torch.device('cpu'))

    pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang, val_pos_loader, val_neg_loader,
                                            mode_declarations, bk_clauses, no_xil=args.no_xil)  # torch.device('cpu'))

    return clause_generator, pi_clause_generator, FC


def train_and_eval(args, pm_prediction_dict, val_pos_loader, val_neg_loader, writer, rtpt, exp_output_path):
    NSFR = None
    FC = None
    val_pos = pm_prediction_dict["val_pos"].to(args.device)
    val_neg = pm_prediction_dict["val_neg"].to(args.device)
    lang, init_clauses, bk_clauses, pi_clauses, bk, atoms = get_lang(args.lark_path, args.lang_base_path,
                                                                     args.dataset_type, args.dataset)

    clauses = update_initial_clauses(init_clauses, args.n_obj)
    bs_clauses = []
    # loop for predicate invention
    for i in range(args.pi_epochs):
        init_clauses = update_initial_clauses(init_clauses, args.n_obj)
        # pi_clauses = []
        # get models
        clause_generator, pi_clause_generator, FC = get_models(args, lang, val_pos_loader, val_neg_loader,
                                                               init_clauses, bk_clauses, pi_clauses, atoms, bk)
        # generate clauses # time-consuming code
        bs_clauses, p_scores_list, thbf = clause_generator.generate(clauses, val_pos, val_neg, pi_clauses,
                                                                    T_beam=args.t_beam, N_beam=args.n_beam,
                                                                    N_max=args.n_max)

        if thbf:
            break

        if args.no_pi:
            clauses = bs_clauses
        else:
            # invent new predicate and generate pi clauses
            new_pi_clauses = pi_clause_generator.generate(bs_clauses, val_pos, val_neg)
            # add new predicates
            pi_clauses += new_pi_clauses
            bk_clauses += new_pi_clauses
            lang = pi_clause_generator.lang
            atoms = logic_utils.get_atoms(lang)

            # clauses = bs_clauses + pi_clauses

    clauses = bs_clauses + pi_clauses
    NSFR = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, pi_clauses, FC, train=True)
    nsfr_loss_list = train_nsfr(args, NSFR, pm_prediction_dict, writer, rtpt, exp_output_path)
    return NSFR


def main(n):
    args = get_args()
    if args.dataset_type == 'kandinsky':
        if args.small_data:
            name = str(Path("small_KP") / f"aILP_{args.dataset}_{str(n)}")
        else:
            name = str(Path("KP") / f"aILP_{args.dataset}_{str(n)}")
    else:
        if not args.no_xil:
            name = str(Path('CH') / Path(f"/aILP_{args.dataset}_{str(n)}"))
        else:
            name = str(Path('CH') / f"aILP-noXIL_{args.dataset}_{str(n)}")
    print('args ', args)
    if args.no_cuda:
        args.device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cuda:' + str(args.device))

    print('device: ', args.device)

    if args.no_pi:
        args.pi_epochs = 1

    # run_name = 'predict/' + args.dataset
    writer = SummaryWriter(str(config.root / "runs" / name), purge_step=0)

    # Create RTPT object
    rtpt = RTPT(name_initials='Jing', experiment_name=name,
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # get torch data loader
    train_loader, val_loader, test_loader = get_data_loader(args)

    train_pos_loader, val_pos_loader, test_pos_loader = get_data_pos_loader(args)
    train_neg_loader, val_neg_loader, test_neg_loader = nsfr_utils.get_data_neg_loader(args)

    pm_prediction_dict = get_perception_predictions(args, val_pos_loader, val_neg_loader,
                                                    train_pos_loader, train_neg_loader,
                                                    test_pos_loader, test_neg_loader)
    #####train_pos_loader, val_pos_loader, test_pos_loader = get_data_loader(args)

    # load logical representations
    lark_path = str(config.root / 'src' / 'lark' / 'exp.lark')
    args.lark_path = lark_path
    lang_base_path = config.root / 'data' / 'lang'
    args.lang_base_path = lang_base_path

    exp_output_path = config.buffer_path / args.dataset
    if not os.path.exists(exp_output_path):
        os.mkdir(exp_output_path)

    # main program
    NSFR = train_and_eval(args, pm_prediction_dict, val_pos_loader, val_neg_loader, writer, rtpt, exp_output_path)
    final_evaluation(NSFR, pm_prediction_dict, args)
    # update PI
    # PI = pi_utils.get_pi_model(args, lang, pi_clauses, atoms, bk, bk_clauses, device=device)
    # params_pi = PI.get_params()
    # optimizer_pi = torch.optim.RMSprop(params_pi, lr=args.lr)
    # # optimizer = torch.optim.Adam(params, lr=args.lr)
    # pi_loss_list = train_pi(args, PI, optimizer_pi, train_loader, val_loader, test_loader, device, writer, rtpt)

    # final evaluation


if __name__ == "__main__":
    main(n=0)
