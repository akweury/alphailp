import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from tqdm import tqdm
import datetime

import pi_utils
from nsfr_utils import get_prob, get_nsfr_model, update_initial_clauses
import logic_utils
from logic_utils import get_lang
from mode_declaration import get_mode_declarations
from clause_generator import ClauseGenerator, PIClauseGenerator
import facts_converter
from percept import YOLOPerceptionModule, FCNNPerceptionModule
from valuation import YOLOValuationModule, PIValuationModule, FCNNValuationModule
import log_utils, file_utils

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def ilp_predict(NSFR, pos_pred, neg_pred, args, th=None, split='train'):
    predicted_list = []
    target_list = []
    count = 0

    pm_pred = torch.cat((pos_pred, neg_pred), dim=0)
    train_label = torch.zeros(len(pm_pred))
    train_label[:len(pos_pred)] = 1.0

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


def train_nsfr(args, NSFR, pm_prediction_dict, rtpt, exp_output_path):
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

            acc_val, rec_val, th_val = ilp_predict(NSFR, pm_prediction_dict['val_pos'],
                                                   pm_prediction_dict['val_neg'], args, th=0.33, split='val')
            # writer.add_scalar("metric/val_acc", acc_val, global_step=epoch)
            log_utils.add_lines(f"acc_val:{acc_val} ", args.log_file)
            log_utils.add_lines("Predi$\alpha$ILPcting on training data set...", args.log_file)

            acc, rec, th = ilp_predict(NSFR, pm_prediction_dict['train_pos'],
                                       pm_prediction_dict['train_neg'], args, th=th_val, split='train')
            # writer.add_scalar("metric/train_acc", acc, global_step=epoch)
            log_utils.add_lines(f"acc_train: {acc}", args.log_file)
            log_utils.add_lines(f"Predicting on test data set...", args.log_file)

            acc, rec, th = ilp_predict(NSFR, pm_prediction_dict['test_pos'],
                                       pm_prediction_dict['test_neg'], args, th=th_val, split='train')
            # writer.add_scalar("metric/test_acc", acc, global_step=epoch)
            log_utils.add_lines(f"acc_test: {acc}", args.log_file)

    return loss


def get_models(args, lang, val_pos_loader, val_neg_loader,
               clauses, pi_clauses, atoms, obj_n):
    if args.dataset_type == "kandinsky":
        PM = YOLOPerceptionModule(e=args.e, d=11, device=args.device)
        VM = YOLOValuationModule(lang=lang, device=args.device, dataset=args.dataset)
    elif args.dataset_type == "hide":
        PM = FCNNPerceptionModule(e=args.e, d=8, device=args.device)
        VM = FCNNValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
    else:
        raise ValueError
    PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
    FC = facts_converter.FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
                                        pi_valuation_module=PI_VM, device=args.device)
    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, pi_clauses, FC)
    PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)

    mode_declarations = get_mode_declarations(args, lang, obj_n)
    clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, val_pos_loader, val_neg_loader,
                                       mode_declarations, no_xil=args.no_xil)  # torch.device('cpu'))

    pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang, val_pos_loader, val_neg_loader,
                                            mode_declarations, no_xil=args.no_xil)  # torch.device('cpu'))

    return clause_generator, pi_clause_generator, FC


def extend_clauses(args, clause_generator, init_clauses, pi_c, max_c, clauses):
    log_utils.add_lines(f"\n=== beam search iteration {args.iteration}/{args.max_step} ===", args.log_file)
    # generate clauses # time-consuming code
    bs_clauses, max_clause, current_step, args = clause_generator.clause_extension(init_clauses, pi_c, args,
                                                                                   max_c)
    if len(bs_clauses) == 0:
        args.is_done = True
    elif len(bs_clauses) > 0 and bs_clauses[0][1][2] == 1.0:
        log_utils.add_lines(f"found sufficient and necessary clause.", args.log_file)
        clauses = logic_utils.extract_clauses_from_bs_clauses([bs_clauses[0]], "sn", args)
        args.is_done = True
        # break
    elif len(bs_clauses) > 0 and bs_clauses[0][1][2] > args.sn_th:
        log_utils.add_lines(f"found quasi-sufficient and necessary clause.", args.log_file)
        clauses = logic_utils.extract_clauses_from_bs_clauses([bs_clauses[0]], "sn_good", args)
        args.is_done = True
        # break
    else:
        if args.pi_top == 0:
            clauses += logic_utils.top_select(bs_clauses, args)
        elif args.iteration == args.max_step:
            clauses += logic_utils.extract_clauses_from_max_clause(bs_clauses, args)
        elif max_clause[1] is not None:
            clauses += logic_utils.extract_clauses_from_max_clause(max_clause[1], args)
        else:
            raise ValueError

    return bs_clauses, clauses, args


def invent_predicates(args, clauses, bs_clauses, pi_clause_generator, new_c, new_p, neural_preds):
    args.no_new_preds = True
    p_new_with_score = []
    lang = pi_clause_generator.lang
    atoms = logic_utils.get_atoms(lang)
    if args.no_pi:
        clauses += logic_utils.extract_clauses_from_bs_clauses(bs_clauses, "clause", args)
    elif args.pi_top > 0:
        # invent new predicate and generate pi clauses
        new_c, new_p, p_new_with_score, is_done = pi_clause_generator.invent_predicate(bs_clauses, new_c, args,
                                                                                       neural_preds, new_p)
        new_pred_num = len(pi_clause_generator.lang.invented_preds) - args.invented_pred_num
        args.invented_pred_num = len(pi_clause_generator.lang.invented_preds)
        if new_pred_num > 0:
            # add new predicates
            args.no_new_preds = False
            lang = pi_clause_generator.lang
            atoms = logic_utils.get_atoms(lang)
            for c in new_c:
                if c not in clauses:
                    clauses.append(c)
    return clauses, new_c, new_p, args, atoms, lang, p_new_with_score


def train_and_eval(args, pm_prediction_dict, val_pos_loader, val_neg_loader, rtpt, exp_output_path):
    FC = None
    invented_preds = []
    all_pi_clauses = []
    clauses = []

    for neural_pred_i in range(len(args.neural_preds)):
        clauses = []
        # load language module
        lang, full_init_clauses, atoms = get_lang(args)
        init_clauses = update_initial_clauses(full_init_clauses, args.n_obj)
        # update language with neural predicate: shape/color/dir/dist
        p_inv_with_scores = []

        if (neural_pred_i < len(args.neural_preds) - 1):
            lang.preds = lang.preds[:2]
            lang.invented_preds = []
            lang.preds.append(args.neural_preds[neural_pred_i][0])
            pi_clauses = []
            pi_p = []
        else:
            print('last round')
            lang.preds = lang.preds[:2] + args.neural_preds[-1]
            lang.invented_preds = invented_preds
            pi_clauses = all_pi_clauses
            pi_p = invented_preds

        atoms = logic_utils.get_atoms(lang)

        args.is_done = False
        args.iteration = 0
        max_clause = [0.0, None]
        args.no_new_preds = False
        args.last_refs = init_clauses
        clause_generator, pi_clause_generator, FC = get_models(args, lang, val_pos_loader, val_neg_loader,
                                                               init_clauses, pi_clauses, atoms, args.n_obj)

        while args.iteration < args.max_step and not args.is_done:
            bs_clauses, clauses, args = extend_clauses(args, clause_generator,
                                                       init_clauses, pi_clauses, max_clause, clauses)
            clauses, pi_clauses, pi_p, args, atoms, lang, p_new_scores = invent_predicates(args, clauses, bs_clauses,
                                                                                           pi_clause_generator,
                                                                                           pi_clauses, pi_p,
                                                                                           args.neural_preds[
                                                                                               neural_pred_i])
            p_inv_with_scores += p_new_scores
            atoms = logic_utils.get_atoms(lang)
            clause_generator, pi_clause_generator, FC = get_models(args, lang, val_pos_loader, val_neg_loader,
                                                                   init_clauses, pi_clauses, atoms, args.n_obj)
            args.iteration += 1

        p_inv_best = sorted(p_inv_with_scores, key=lambda x: x[1][2], reverse=True)
        p_inv_th = p_inv_best[args.pi_top][1][2]
        p_inv_best = [p_inv for p_inv in p_inv_best if p_inv[1][2] >= p_inv_th]
        p_inv_best = logic_utils.extract_clauses_from_bs_clauses(p_inv_best, "best inv clause", args)

        for new_p in p_inv_best:
            if new_p not in invented_preds:
                invented_preds.append(new_p)
        for new_c in pi_clauses:
            if new_c not in all_pi_clauses and new_c.head.pred in p_inv_best:
                all_pi_clauses.append(new_c)

        if args.found_ns:
            break

    if len(clauses) > 0:
        for c in clauses:
            log_utils.add_lines(f"(final NSFR clause) {c}", args.log_file)
    NSFR = get_nsfr_model(args, lang, clauses, atoms, all_pi_clauses, FC, train=True)
    train_nsfr(args, NSFR, pm_prediction_dict, rtpt, exp_output_path)
    return NSFR


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
