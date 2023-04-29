from sklearn.metrics import accuracy_score, recall_score, roc_curve
from tqdm import tqdm
import datetime

from nsfr_utils import get_prob, get_nsfr_model
from pi_utils import *
import aitk

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

# def get_models(args, lang, clauses, pi_clauses, atoms):
#     obj_n = args.e
#     if args.dataset_type == "kandinsky":
#         PM = YOLOPerceptionModule(e=args.e, d=11, device=args.device)
#         VM = YOLOValuationModule(lang=lang, device=args.device, dataset=args.dataset)
#     elif args.dataset_type == "hide":
#         PM = FCNNPerceptionModule(e=args.e, d=8, device=args.device)
#         VM = FCNNValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
#     else:
#         raise ValueError
#     PI_VM = PIValuationModule(lang=lang, device=args.device, dataset=args.dataset, dataset_type=args.dataset_type)
#     FC = facts_converter.FactsConverter(lang=lang, perception_module=PM, valuation_module=VM,
#                                         pi_valuation_module=PI_VM, device=args.device)
#     # Neuro-Symbolic Forward Reasoner for clause generation
#     NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, pi_clauses, FC)
#     PI_cgen = pi_utils.get_pi_model(args, lang, clauses, atoms, pi_clauses, FC)
#
#     mode_declarations = get_mode_declarations(args, lang)
#     clause_generator = ClauseGenerator(args, NSFR_cgen, PI_cgen, lang, mode_declarations,
#                                        no_xil=args.no_xil)  # torch.device('cpu'))
#
#     pi_clause_generator = PIClauseGenerator(args, NSFR_cgen, PI_cgen, lang, no_xil=args.no_xil)  # torch.device('cpu'))
#
#     return clause_generator, pi_clause_generator, FC
