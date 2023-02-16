import os.path
from operator import itemgetter
import random
from nsfr_utils import update_nsfr_clauses, get_prob, get_nsfr_model
# from eval_clause import EvalInferModule
from refinement import RefinementGenerator
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pi_utils import get_pi_model
import chart_utils
from percept import YOLOPerceptionModule
import config
import logic_utils
from fol.language import Language, DataType
import fol.logic as logic


def check_accuracy(clause_scores_full,pair_num):
    accuracy = clause_scores_full[:, 1] / pair_num

    return accuracy


class ClauseGenerator(object):
    """
    clause generator by refinement and beam search
    Parameters
    ----------
    ilp_problem : .ilp_problem.ILPProblem
    infer_step : int
        number of steps in forward inference
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, args, NSFR, PI, lang, pos_data_loader, neg_data_loader, mode_declarations, bk_clauses,
                 no_xil=False):
        self.args = args
        self.NSFR = NSFR
        self.PI = PI
        self.lang = lang
        self.mode_declarations = mode_declarations
        self.bk_clauses = bk_clauses
        self.device = args.device
        self.no_xil = no_xil
        self.rgen = RefinementGenerator(lang=lang, mode_declarations=mode_declarations)
        self.pos_loader = pos_data_loader
        self.neg_loader = neg_data_loader
        self.bce_loss = torch.nn.BCELoss()

        # self.labels = torch.cat([
        #    torch.ones((len(self.ilp_problem.pos), )),
        # ], dim=0).to(device)

    def _is_valid(self, clause):
        obj_num = len([b for b in clause.body if b.pred.name == 'in'])
        attr_body = [b for b in clause.body if b.pred.name != 'in']
        attr_vars = []
        for b in attr_body:
            dtypes = b.pred.dtypes
            for i, term in enumerate(b.terms):
                if dtypes[i].name == 'object' and term.is_var():
                    attr_vars.append(term)

        attr_vars = list(set(attr_vars))

        # print(clause, obj_num, attr_vars)
        return obj_num == len(attr_vars)  # or len(attr_body) == 0

    def _cf0(self, clause):
        """Confounded rule for CLEVR-Hans.
        not gray
        """
        for bi in clause.body:
            if bi.pred.name == 'color' and str(bi.terms[-1]) == 'gray':
                return True
        return False

    def _cf1(self, clause):
        """not metal sphere.
        """
        for bi in clause.body:
            for bj in clause.body:
                if bi.pred.name == 'material' and str(bi.terms[-1]) == 'gray':
                    if bj.pred.name == 'shape' and str(bj.terms[-1]) == 'sphere':
                        return True
        return False

    def _is_confounded(self, clause):
        if self.no_xil:
            return False
        if self.args.dataset_type == 'kandinsky':
            return False
        else:
            if self.args.dataset == 'clevr-hans0':
                return self._cf0(clause)
            elif self.args.dataset == 'clevr-hans1':
                return self._cf1(clause)
            else:
                return False

    def generate(self, C_0, pos_pred, neg_pred, pi_clauses, gen_mode='beam', T_beam=7, N_beam=20, N_max=100):
        """
        call clause generation function with or without beam-searching
        Inputs
        ------
        C_0 : Set[.logic.Clause]
            a set of initial clauses
        gen_mode : string
            a generation mode
            'beam' - with beam-searching
            'naive' - without beam-searching
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated
        Returns
        -------
        C : Set[.logic.Clause]
            set of generated clauses
        """
        if gen_mode == 'beam':
            beam_search_clauses, p_scores_list, thbf = self.beam_search(C_0, pos_pred, neg_pred, pi_clauses,
                                                                        T_beam=T_beam, N_beam=N_beam, N_max=N_max)
            return beam_search_clauses, p_scores_list, thbf
        elif gen_mode == 'naive':
            return self.naive(C_0, T_beam=T_beam, N_max=N_max)

    def beam_search_clause(self, init_clause, pos_pred, neg_pred, T_beam=7, N_beam=20, N_max=100, th=0.98):
        """
        perform beam-searching from a clause
        Inputs
        ------
        clause : Clause
            initial clause
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated
        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses
        """
        step = 0
        init_step = 0
        B = [init_clause]
        C = set()
        C_dic = {}
        B_ = []
        lang = self.lang

        while step < T_beam:
            # print('Beam step: ', str(step),  'Beam: ', len(B))
            B_new = {}
            refs = []
            for c in B:
                refs_i = self.rgen.refinement_clause(c)
                # remove invalid clauses
                ###refs_i = [x for x in refs_i if self._is_valid(x)]
                # remove already appeared refs
                refs_i = list(set(refs_i).difference(set(B_)))
                B_.extend(refs_i)
                refs.extend(refs_i)
                if self._is_valid(c) and not self._is_confounded(c):
                    C = C.union(set([c]))
                    print("Added: ", c)

            print('Evaluating ', len(refs), 'generated clauses.')
            # evaluate clauses, it should consider both positive images as well as negative images.
            loss_list = self.eval_clauses(refs, pos_pred, neg_pred)
            for i, ref in enumerate(refs):
                # check duplication
                if not self.is_in_beam(B_new, ref, loss_list[i]):
                    B_new[ref] = loss_list[i]
                    C_dic[ref] = loss_list[i]

                # if len(C) >= N_max:
                #    break
            B_new_sorted = sorted(B_new.items(), key=lambda x: x[1], reverse=True)

            # top N_beam refiements
            B_new_sorted = B_new_sorted[:N_beam]
            # B_new_sorted = [x for x in B_new_sorted if x[1] > th]
            for x in B_new_sorted:
                print(x[1], x[0])
            B = [x[0] for x in B_new_sorted]
            step += 1
            if len(B) == 0:
                break
            # if len(C) >= N_max:
            #    break
        return C

    # def remove_conflict_clauses(self, clauses):
    #     print("check for conflict clauses...")
    #     non_conflict_clauses = []
    #     for clause in clauses:
    #         is_conflict = False
    #         for i in range(len(clause.body)):
    #             for j in range(i + 1, len(clause.body)):
    #                 if "at_area" in clause.body[i].pred.name and "at_area" in clause.body[j].pred.name:
    #                     if clause.body[i].terms == clause.body[j].terms:
    #                         is_conflict = True
    #                         print(f'conflict clause: {clause}')
    #                         break
    #                     elif self.conflict_pred(clause.body[i].pred.name, clause.body[j].pred.name,
    #                                             list(clause.body[i].terms), list(clause.body[j].terms)):
    #                         is_conflict = True
    #                         print(f'conflict clause: {clause}')
    #                         break
    #             if is_conflict:
    #                 break
    #         if not is_conflict:
    #             non_conflict_clauses.append(clause)
    #
    #     print("end for checking.")
    #     print("========= All non-conflict clauses ==========")
    #     for each in non_conflict_clauses:
    #         print(each)
    #     print("=============================================")
    #
    #     return non_conflict_clauses
    #
    # def conflict_pred(self, p1, p2, t1, t2):
    #     non_confliect_dict = {
    #         "at_area_0": ["at_area_2"],
    #         "at_area_1": ["at_area_3"],
    #         "at_area_2": ["at_area_0"],
    #         "at_area_3": ["at_area_1"],
    #         "at_area_4": ["at_area_6"],
    #         "at_area_5": ["at_area_7"],
    #         "at_area_6": ["at_area_4"],
    #         "at_area_7": ["at_area_5"],
    #     }
    #     if p1 in non_confliect_dict.keys():
    #         if "at_area" in p2 and p2 not in non_confliect_dict[p1]:
    #             if t1[0] == t2[1] and t2[0] == t1[1]:
    #                 return True
    #     return False

    def beam_search_clause_quick(self, init_clause, pos_pred, neg_pred, pi_clauses, T_beam=7, N_beam=20, N_max=100,
                                 th=0.98):

        step = 0
        init_step = 0
        B = [init_clause]
        C = set()
        C_dic = {}
        B_ = []
        eval_pred_names = ['kp']
        p_score = None
        lang = self.lang
        bs_clauses = []
        target_has_been_found = False
        while step < T_beam:
            B_new = {}
            refs = []
            for c in B:
                refs_i = self.rgen.refinement_clause(c)
                # remove invalid clauses
                ###refs_i = [x for x in refs_i if self._is_valid(x)]
                # remove already appeared refs
                refs_i = list(set(refs_i).difference(set(B_)))
                B_.extend(refs_i)
                refs.extend(refs_i)
                if self._is_valid(c) and not self._is_confounded(c):
                    C = C.union(set([c]))
                    print(f"(BS step {step + 1}/{T_beam}) Added: ", c)

            # evaluate clauses, it should consider both positive images as well as negative images.
            refs_non_conflict = logic_utils.remove_conflict_clauses(refs, pi_clauses)

            # for r_i, ref in enumerate(refs_non_conflict):
            #     print(f"(BS step {step + 1}/{T_beam}) Non Conflict clause {r_i + 1}/{len(refs_non_conflict)}: {ref}")
            print('Evaluating ', len(refs_non_conflict), 'generated clauses.')
            # refs_non_conflict = logic_utils.reorder_terms(refs_non_conflict)
            self.NSFR = get_nsfr_model(self.args, self.lang, refs_non_conflict, self.NSFR.atoms,
                                       self.NSFR.bk, self.bk_clauses, pi_clauses, self.NSFR.fc, self.device)

            clause_signs, clause_score_list, clause_scores_full = \
                logic_utils.eval_predicates(self.NSFR, self.args, eval_pred_names, pos_pred, neg_pred)

            # clause_image_scores = self.eval_clauses_scores(refs_non_conflict, pos_pred, neg_pred)

            clause_accuracy = check_accuracy(clause_scores_full, pos_pred.size(0))
            if clause_accuracy.max() == 1.0:
                print("target clause has been found.")
                target_has_been_found = True
                c_indices = [np.argmax(clause_accuracy)]
                for c_i in c_indices:
                    print(refs_non_conflict[c_i])
                break
            else:
                print(f"max clause accuracy: {clause_accuracy.max()}")
            # check for duplication
            non_duplicate_clause_index = []
            non_duplicate_full_scores = []
            for i, ref in enumerate(refs_non_conflict):
                # check duplication
                if not self.is_in_beam(B_new, ref, clause_scores_full[i, :]):
                    B_new[ref] = clause_scores_full[i, :]
                    C_dic[ref] = clause_scores_full[i, :]
                    non_duplicate_clause_index.append(i)
                    non_duplicate_full_scores.append(clause_scores_full[i, :])
            is_plot_4zone = False
            # if is_plot_4zone:
            #     for i, clause in enumerate(B_new):
            #         clause_index = non_duplicate_clause_index[i]
            #         chart_utils.plot_scatter_heat_chart([p_score[clause_index]],
            #                                             config.buffer_path / "img",
            #                                             f"heat_ce_all_{len(B_new)}_{i}",
            #                                             sub_folder=str(step),
            #                                             labels=f"{str(clause) + str(clause_signs[clause_index])}",
            #                                             x_label="positive score", y_label="negative score")
            #
            #         clause_scores_reverse = [p_score[clause_index][1], p_score[clause_index][0]]
            #         chart_utils.plot_scatter_chart([clause_scores_reverse], config.buffer_path / "img",
            #                                        f"scatter_ce_all_{len(B_new)}_{i}",
            #                                        sub_folder=str(step),
            #                                        labels=f"{str(clause) + str(clause_signs[clause_index])}",
            #                                        x_label="positive score", y_label="negative score")
            # B_new_sorted = sorted(B_new.items(), key=lambda x: x[0, 1], reverse=True)
            B_new_good = self.select_good_clauses(B_new)
            #
            # if len(B_new_good) > 10:
            #     B_new_good = random.sample(B_new_good, 10)

            #     B_new_sorted = B_new_sorted[:5]
            # B_new_sorted = [x for x in B_new_sorted if x[1] > th]
            # for x in B_new_sorted:
            # print(f'(BS Clause on Step {step} )' + str(x[1]) + ', ' + str(x[0]))
            B = [x[0] for x in B_new_good]
            step += 1
            if len(B) == 0:
                break

        # C = logic_utils.remove_extended_clauses(C, p_score)

        return C, p_score, target_has_been_found

    def eval_images(self, save_path):

        prop_dim = 11
        # perception model
        pm = YOLOPerceptionModule(e=self.args.e, d=prop_dim, device=self.device)

        # positive image evaluation
        N_data = 0
        pos_eval_res = torch.zeros((self.pos_loader.dataset.__len__(), self.args.e, prop_dim)).to(self.device)
        for i, sample in tqdm(enumerate(self.pos_loader, start=0)):
            imgs, target_set = map(lambda x: x.to(self.device), sample)
            # print(NSFR.clauses)
            img_array = imgs.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
            img_array_int8 = np.uint8(img_array * 255)
            img_pil = Image.fromarray(img_array_int8)
            # img_pil.show()
            N_data += imgs.size(0)
            B = imgs.size(0)
            # C * B * G
            # when evaluate a clause which its body contains invented predicates,
            # the invented predicates shall be evaluated with all the clauses which head contains the predicate.
            res = pm(imgs)
            pos_eval_res[i, :] = res

            # negative image evaluation
        N_data = 0
        neg_eval_res = torch.zeros((self.neg_loader.dataset.__len__(), self.args.e, prop_dim)).to(self.device)
        for i, sample in tqdm(enumerate(self.pos_loader, start=0)):
            imgs, target_set = map(lambda x: x.to(self.device), sample)
            # print(NSFR.clauses)
            img_array = imgs.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
            img_array_int8 = np.uint8(img_array * 255)
            img_pil = Image.fromarray(img_array_int8)
            # img_pil.show()
            N_data += imgs.size(0)
            B = imgs.size(0)
            # C * B * G
            # when evaluate a clause which its body contains invented predicates,
            # the invented predicates shall be evaluated with all the clauses which head contains the predicate.
            res = pm(imgs)
            neg_eval_res[i, :] = res

        # save tensors
        pm_res = {'pos_res': pos_eval_res.detach(),
                  'neg_res': neg_eval_res.detach()}
        torch.save(pm_res, str(save_path))

        return pos_eval_res, neg_eval_res

    def is_in_beam(self, B, clause, scores):
        """If score is the same, same predicates => duplication
        """

        preds = set([clause.head.pred] + [b.pred for b in clause.body])
        c_terms = set(clause.head.terms + [t for b in clause.body for t in b.terms])
        y = False
        for ci, score_i in B.items():
            preds_i = set([ci.head.pred] + [b.pred for b in ci.body])
            ci_terms = set(ci.head.terms + [t for b in ci.body for t in b.terms])
            if preds == preds_i and c_terms == ci_terms:
                y = True
                # print("duplicated: ", clause, ci)
                break
        return y

    def beam_search(self, C_0, pos_pred, neg_pred, pi_clauses, T_beam=7, N_beam=20, N_max=100):
        """
        generate clauses by beam-searching from initial clauses
        Inputs
        ------
        C_0 : Set[.logic.Clause]
            set of initial clauses
        T_beam : int
            number of steps in beam-searching
        N_beam : int
            size of the beam
        N_max : int
            maximum number of clauses to be generated
        Returns
        -------
        C : Set[.logic.Clause]
            a set of generated clauses
        """
        C = set()
        p_scores_list = []
        thbf = False
        for clause in C_0:
            bs_clauses, p_scores, thbf = self.beam_search_clause_quick(clause, pos_pred, neg_pred, pi_clauses, T_beam,
                                                                       N_beam,
                                                                       N_max)
            C = C.union(bs_clauses)
            p_scores_list.append(p_scores)
            # C = C.union(self.beam_search_clause(clause, pos_pred, neg_pred, T_beam, N_beam, N_max))
        C = sorted(list(C))
        print('\n======= BEAM SEARCHED CLAUSES ======')
        for c in C:
            print('(BS Clause) ' + str(c))
        print("====== ", len(C), " clauses are generated!! ======")
        return C, p_scores_list, thbf

    def eval_pi_clauses(self, clauses):
        return None

    def eval_clause_sign(self, p_scores):
        resolution = 2

        p_clauses_signs = []
        for p_score in p_scores.values():
            clause_sign_list = []
            clause_score_list = []
            clause_score_full_list = []
            for clause_image_score in p_score:
                data_map = np.zeros(shape=[resolution, resolution])
                for index in range(len(clause_image_score[0][0])):
                    x_index = int(clause_image_score[0][0, index] * resolution)
                    y_index = int(clause_image_score[1][0, index] * resolution)
                    data_map[x_index, y_index] += 1

                pos_low_neg_low_area = data_map[0, 0]
                pos_high_neg_low_area = data_map[0, 1]
                pos_low_neg_high_area = data_map[1, 0]
                pos_high_neg_high_area = data_map[1, 1]

                # TODO: find a better score evaluation function
                clause_score = pos_high_neg_low_area - pos_high_neg_high_area
                clause_score_list.append(clause_score)
                clause_score_full_list.append(
                    [pos_low_neg_low_area, pos_high_neg_low_area, pos_low_neg_high_area, pos_high_neg_high_area])

                data_map[0, 0] = 0
                if np.max(data_map) == data_map[0, 1] and data_map[0, 1] > data_map[1, 1]:
                    clause_sign_list.append(True)
                else:
                    clause_sign_list.append(False)
            p_clauses_signs.append([clause_sign_list, clause_score_list, clause_score_full_list])
        return p_clauses_signs

    def eval_clauses(self, clauses, pos_pm_res, neg_pm_res):
        C = len(clauses)
        print("Eval clauses: ", len(clauses))
        # update infer module with new clauses
        # NSFR = update_nsfr_clauses(self.NSFR, clauses, self.bk_clauses, self.device)
        pi_clauses = []
        batch_size = self.args.batch_size_bs
        NSFR = get_nsfr_model(self.args, self.lang, clauses, self.NSFR.atoms,
                              self.NSFR.bk, self.bk_clauses, pi_clauses, self.NSFR.fc, self.device)
        # TODO: Compute loss for validation data , score is bce loss
        pos_img_num = self.pos_loader.dataset.__len__()
        neg_img_num = self.neg_loader.dataset.__len__()

        positive_score = torch.zeros((pos_img_num, C)).to(self.device)
        negative_score = torch.zeros((neg_img_num, C)).to(self.device)

        for i in range(self.pos_loader.dataset.__len__()):
            V_T_list = NSFR.clause_eval_quick(pos_pm_res[i].unsqueeze(0)).detach()
            C_score = torch.zeros((C, batch_size)).to(self.device)
            for j, V_T in enumerate(V_T_list):
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                C_score[j] = predicted
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
            C_score = C_score.sum(dim=1)
            positive_score[i, :] = C_score
        a = positive_score.detach().numpy()

        for i in range(self.neg_loader.dataset.__len__()):
            V_T_list = NSFR.clause_eval_quick(neg_pm_res[i].unsqueeze(0)).detach()
            C_score = torch.zeros((C, batch_size)).to(self.device)
            for j, V_T in enumerate(V_T_list):
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                C_score[j] = predicted
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
            C_score = C_score.sum(dim=1)
            negative_score[i] = C_score
        b = negative_score.detach().numpy()

        # positive_clauses = []
        all_clauses_scores = []
        for c_index in range(positive_score.shape[1]):
            clause_scores = [negative_score[:, c_index], positive_score[:, c_index]]
            clause_sign, clause_score, clause_score_full = self.eval_clause_sign(clause_scores)
            all_clauses_scores.append(clause_score)
            # if clause_sign:
            #     positive_clauses.append(clauses[c_index])
            # plot the clause evaluation

            # clause_scores_reverse = [positive_score[:, c_index], negative_score[:, c_index]]
            # chart_utils.plot_scatter_chart([clause_scores_reverse], config.buffer_path / "img",
            #                                f"scatter_ce_all_{len(clauses)}_{c_index}",
            #                                labels=f"{str(clauses[c_index]) + str(clause_sign)}",
            #                                x_label="positive score", y_label="negative score")

        return all_clauses_scores

    def eval_clauses_scores(self, clauses, pos_pm_res, neg_pm_res):
        C = len(clauses)
        print(f"Eval clauses on {len(clauses)} images...")
        # update infer module with new clauses
        # NSFR = update_nsfr_clauses(self.NSFR, clauses, self.bk_clauses, self.device)
        pi_clauses = []
        batch_size = self.args.batch_size_bs
        NSFR = get_nsfr_model(self.args, self.lang, clauses, self.NSFR.atoms,
                              self.NSFR.bk, self.bk_clauses, pi_clauses, self.NSFR.fc, self.device)
        # TODO: Compute loss for validation data , score is bce loss
        pos_img_num = self.pos_loader.dataset.__len__()
        neg_img_num = self.neg_loader.dataset.__len__()

        positive_score = torch.zeros((pos_img_num, C)).to(self.device)
        negative_score = torch.zeros((neg_img_num, C)).to(self.device)

        for i in range(self.pos_loader.dataset.__len__()):
            V_T_list = NSFR.clause_eval_quick(pos_pm_res[i].unsqueeze(0)).detach()
            C_score = torch.zeros((C, batch_size)).to(self.device)
            for j, V_T in enumerate(V_T_list):
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                C_score[j] = predicted
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
            C_score = C_score.sum(dim=1)
            positive_score[i, :] = C_score
        a = positive_score.detach().to("cpu").numpy()

        for i in range(self.neg_loader.dataset.__len__()):
            V_T_list = NSFR.clause_eval_quick(neg_pm_res[i].unsqueeze(0)).detach()
            C_score = torch.zeros((C, batch_size)).to(self.device)
            for j, V_T in enumerate(V_T_list):
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                C_score[j] = predicted
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
            C_score = C_score.sum(dim=1)
            negative_score[i] = C_score
        b = negative_score.detach().to("cpu").numpy()

        positive_clauses = []
        all_clause_scores = []
        for c_index in range(positive_score.shape[1]):
            clause_scores = [negative_score[:, c_index], positive_score[:, c_index]]
            all_clause_scores.append(clause_scores)
        return all_clause_scores

    def select_good_clauses(self, B_new_sorted):
        good_clauses = []

        for clause, scores in B_new_sorted.items():
            last_3 = scores[1:]
            if torch.max(last_3, dim=-1)[0] == last_3[0] and last_3[0] > last_3[2]:
                good_clauses.append((clause, scores))
            # if last_3[0] + last_3[2] == self.pos_loader.dataset.__len__():
            #     good_clauses.append((clause, scores))
        return good_clauses


class PIClauseGenerator(object):
    """
    clause generator by refinement and beam search
    Parameters
    ----------
    ilp_problem : .ilp_problem.ILPProblem
    infer_step : int
        number of steps in forward inference
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, args, NSFR, PI, lang, pos_data_loader, neg_data_loader, mode_declarations, bk_clauses,
                 no_xil=False):
        self.args = args
        self.lang = lang
        self.NSFR = NSFR
        self.PI = PI
        self.bk_clauses = bk_clauses
        self.device = args.device
        self.pos_loader = pos_data_loader
        self.neg_loader = neg_data_loader

    def generate(self, beam_search_clauses, pos_pred, neg_pred):

        # evaluate for all the clauses
        # clause_image_scores = self.eval_multi_clauses(beam_search_clauses, pos_pred, neg_pred)  # time-consuming line
        # clause_signs, clause_scores, clause_scores_full = self.eval_clause_sign(clause_image_scores)

        # clause_candidates = logic_utils.eval_clause_clusters(clause_clusters, p_scores_list)

        # generate new clauses
        new_predicates = self.generate_new_predicate(beam_search_clauses)
        new_predicates = logic_utils.remove_unaligned_predicates(new_predicates)
        new_predicates = logic_utils.remove_duplicate_predicates(new_predicates)
        # convert to strings
        new_clauses_str_list = self.generate_new_clauses_str_list(new_predicates)

        # convert clauses from strings to objects
        pi_languages = logic_utils.get_pi_clauses_objs(self.args, self.lang, new_clauses_str_list,
                                                       new_predicates)

        pos_pred = pos_pred.to(self.args.device)
        neg_pred = neg_pred.to(self.args.device)
        # generate pi clauses
        passed_pi_languages = self.eval_pi_language(beam_search_clauses, pi_languages, pos_pred, neg_pred)
        # passed_pi_languages = passed_pi_languages[:5]

        passed_pi_clauses, passed_pi_predicates = self.extract_pi(passed_pi_languages)

        print("====== ", len(passed_pi_predicates), " PI predicates are generated!! ======")
        for pred in passed_pi_predicates:
            print(pred)
        print("====== ", len(passed_pi_clauses), " PI clauses are generated!! ======")
        for c in passed_pi_clauses:
            print(c)

        self.lang.invented_preds = passed_pi_predicates

        return passed_pi_clauses

    def eval_multi_clauses(self, clauses, pos_pred, neg_pred):

        C = len(clauses)
        print("Eval clauses: ", len(clauses))
        # update infer module with new clauses
        # NSFR = update_nsfr_clauses(self.NSFR, clauses, self.bk_clauses, self.device)
        pi_clauses = []
        NSFR = get_nsfr_model(self.args, self.lang, clauses, self.NSFR.atoms, self.NSFR.bk, self.bk_clauses, pi_clauses,
                              self.NSFR.fc, self.device)

        batch_size = self.args.batch_size_bs
        pos_img_num = self.pos_loader.dataset.__len__()
        neg_img_num = self.neg_loader.dataset.__len__()
        score_positive = torch.zeros((pos_img_num, C)).to(self.device)
        score_negative = torch.zeros((neg_img_num, C)).to(self.device)
        N_data = 0
        # List(C * B * G)

        # positive image loop
        for image_index in range(self.pos_loader.dataset.__len__()):
            V_T_list = NSFR.clause_eval_quick(pos_pred[image_index].unsqueeze(0)).detach()
            C_score = torch.zeros((C, batch_size)).to(self.device)

            # clause loop
            for clause_index, V_T in enumerate(V_T_list):
                # TODO: eval inv pred
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                C_score[clause_index] = predicted
            # sum over positive prob
            score_positive[image_index, :] = C_score.squeeze(1)

        # negative image loop
        for image_index in range(self.neg_loader.dataset.__len__()):
            V_T_list = NSFR.clause_eval_quick(neg_pred[image_index].unsqueeze(0)).detach()
            C_score = torch.zeros((C, batch_size)).to(self.device)
            for clause_index, V_T in enumerate(V_T_list):
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                C_score[clause_index] = predicted
                # C
                # C_score = PI.clause_eval(C_score)
                # sum over positive prob
            score_negative[image_index, :] = C_score.squeeze(1)

        # all_clauses_scores = []
        # for c_index in range(score_positive.shape[1]):
        #     clause_scores = [score_negative[:, c_index], score_positive[:, c_index]]
        #     clause_sign, clause_score, clause_score_full = self.eval_clause_sign(clause_scores)
        #     all_clauses_scores.append(clause_score)
        all_clause_scores = []
        for c_index in range(score_positive.shape[1]):
            clause_scores = [score_negative[:, c_index], score_positive[:, c_index]]
            all_clause_scores.append(clause_scores)

        return all_clause_scores

    # def remove_conflict_clauses(self, clauses):
    #     print("check for conflict clauses...")
    #     non_conflict_clauses = []
    #     for clause in clauses:
    #         is_conflict = False
    #         for i in range(len(clause.body)):
    #             for j in range(i + 1, len(clause.body)):
    #                 if clause.body[i].terms == clause.body[j].terms:
    #                     is_conflict = True
    #                     print(f'conflict clause: {clause}')
    #                     break
    #                 elif self.conflict_pred(clause.body[i].pred.name, clause.body[j].pred.name,
    #                                         list(clause.body[i].terms), list(clause.body[j].terms)):
    #                     is_conflict = True
    #                     print(f'conflict clause: {clause}')
    #                     break
    #             if is_conflict:
    #                 break
    #         if not is_conflict:
    #             non_conflict_clauses.append(clause)
    #
    #     print("end for checking.")
    #     print("========= All non-conflict clauses ==========")
    #     for each in non_conflict_clauses:
    #         print(each)
    #     print("=============================================")
    #
    #     return non_conflict_clauses
    #
    # def conflict_pred(self, p1, p2, t1, t2):
    #     non_confliect_dict = {
    #         "at_area_0": ["at_area_2"],
    #         "at_area_1": ["at_area_3"],
    #         "at_area_2": ["at_area_0"],
    #         "at_area_3": ["at_area_1"],
    #         "at_area_4": ["at_area_6"],
    #         "at_area_5": ["at_area_7"],
    #         "at_area_6": ["at_area_4"],
    #         "at_area_7": ["at_area_5"],
    #     }
    #     if p1 in non_confliect_dict.keys():
    #         if "at_area" in p2 and p2 not in non_confliect_dict[p1]:
    #             if t1[0] == t2[1] and t2[0] == t1[1]:
    #                 return True
    #     return False

    def eval_clause_sign(self, clause_image_scores):
        # resolution = 2
        # data_map = np.zeros(shape=[resolution, resolution])
        # for index in range(len(clause_scores[0])):
        #     x_index = int(clause_scores[0][index] * resolution)
        #     y_index = int(clause_scores[1][index] * resolution)
        #     data_map[x_index, y_index] += 1
        #
        # if np.max(data_map) == data_map[0, 1]:
        #     return True
        #
        # return False
        resolution = 2

        clause_sign_list = []
        clause_score_list = []
        clause_score_full_list = []
        for clause_image_score in clause_image_scores:
            data_map = np.zeros(shape=[resolution, resolution])
            for index in range(len(clause_image_score[0])):
                x_index = int(clause_image_score[0][index] * resolution)
                y_index = int(clause_image_score[1][index] * resolution)
                data_map[x_index, y_index] += 1

            pos_low_neg_low_area = data_map[0, 0]
            pos_high_neg_low_area = data_map[0, 1]
            pos_low_neg_high_area = data_map[1, 0]
            pos_high_neg_high_area = data_map[1, 1]

            # TODO: find a better score evaluation function
            clause_score = pos_high_neg_low_area - pos_high_neg_high_area
            clause_score_list.append(clause_score)
            clause_score_full_list.append(
                [pos_low_neg_low_area, pos_high_neg_low_area, pos_low_neg_high_area, pos_high_neg_high_area])

            data_map[0, 0] = 0
            if np.max(data_map) == data_map[0, 1] and data_map[0, 1] > data_map[1, 1]:
                clause_sign_list.append(True)
            else:
                clause_sign_list.append(False)

        return clause_sign_list, clause_score_list, clause_score_full_list

    def generate_new_predicate(self, beam_search_clauses):
        new_predicate = None
        # positive_clauses_exchange = [(c[1], c[0]) for c in positive_clauses]
        # no_hn_ = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0 and c[0][3] == 0]
        # no_hnlp = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0]
        # score clauses properly

        new_predicates = []

        # simple predicates
        # # TODO: try to remove this kind of pi
        # for bs_index, bs_clause in enumerate(beam_search_clauses):
        #     if len(bs_clause.body) <= 3:
        #         continue
        #     dtypes = [DataType(dt) for dt in ["object", "object"]]
        #
        #     # self invent
        #     new_predicate = self.lang.get_new_invented_predicate(arity=2, pi_dtypes=dtypes)
        #     non_inv_pred = False
        #     for b in bs_clause.body:
        #         if "inv_pred" in b.pred.name:
        #             non_inv_pred = True
        #             continue
        #     if non_inv_pred:
        #         continue
        #     new_predicate.body = [bs_clause.body]
        #     new_predicates.append(new_predicate)

            # # symmetric invent
            # new_sym_predicate = self.lang.get_new_invented_predicate(arity=2, pi_dtypes=dtypes)
            # symmetry_bodies = []
            # for b in bs_clause.body:
            #     if "at_area" in b.pred.name:
            #         sym_b_pred = logic.Predicate(b.pred.name, 2, dtypes)
            #         sym_b_terms = (logic.Var(b.terms[1].name), logic.Var(b.terms[0].name))
            #         sym_b = logic.Atom(sym_b_pred, sym_b_terms)
            #         symmetry_bodies.append(sym_b)
            #     else:
            #         symmetry_bodies.append(b)
            # new_sym_predicate.body = [symmetry_bodies]
            # new_predicates.append(new_sym_predicate)

            # # cluster invent
            # new_clu_predicate = self.lang.get_new_invented_predicate(arity=2, pi_dtypes=dtypes)
            # new_clu_predicate.body = new_sym_predicate.body + new_predicate.body
            # new_predicates.append(new_clu_predicate)

        # cluster predicates
        pi_clause_clusters = logic_utils.search_independent_clauses(beam_search_clauses)

        for pi_index, clause_cluster in enumerate(pi_clause_clusters):
            dtypes = [DataType(dt) for dt in ["object", "object"]]
            new_predicate = self.lang.get_new_invented_predicate(arity=2, pi_dtypes=dtypes)
            new_predicate.body = [clause.body for clause in clause_cluster]
            new_predicates.append(new_predicate)

            # symmetric invent
            new_sym_predicate = self.lang.get_new_invented_predicate(arity=2, pi_dtypes=dtypes)
            symmetry_bodies = []
            for b_list in new_predicate.body:
                symmetry_b = []
                for b in b_list:
                    if "at_area" in b.pred.name:
                        sym_b_pred = logic.Predicate(b.pred.name, 2, dtypes)
                        sym_b_terms = (logic.Var(b.terms[1].name), logic.Var(b.terms[0].name))
                        sym_b = logic.Atom(sym_b_pred, sym_b_terms)
                        symmetry_b.append(sym_b)
                    else:
                        symmetry_b.append(b)
                symmetry_bodies.append(symmetry_b)
            new_sym_predicate.body = symmetry_bodies + new_predicate.body
            new_predicates.append(new_sym_predicate)

        return new_predicates

    def generate_new_clauses_str_list(self, new_predicates):
        pi_str_lists = []

        for new_predicate in new_predicates:
            single_pi_str_list = []
            # single_pi_str_list.append(f"kp(X):-" + new_predicate.name + "(O1,O2),in(O1,X),in(O2,X).")
            head_args = "(O1,O2)" if new_predicate.arity == 2 else "(X)"
            head = new_predicate.name + head_args + ":-"
            for body in new_predicate.body:
                body_str = ""
                for atom_index in range(len(body)):
                    atom_str = str(body[atom_index])
                    # atom_str = atom_str.replace("O1", "A")
                    # atom_str = atom_str.replace("O2", "B")
                    end_str = "." if atom_index == len(body) - 1 else ","
                    body_str += atom_str + end_str
                new_clause = head + body_str
                single_pi_str_list.append(new_clause)
            pi_str_lists.append(single_pi_str_list)
        # for p_i, p_list in enumerate(pi_str_lists):
            # for p in p_list:
            #     print(f"{p_i}/{len(pi_str_lists)} Invented Predicate: {p}")
        return pi_str_lists

    # def eval_pi_clause_single(self, lang, atoms, clauses, pi_clauses, pos_pred, neg_pred):
    #
    #     NSFR = get_nsfr_model(self.args, lang, clauses, atoms,
    #                           self.NSFR.bk, self.bk_clauses, pi_clauses, self.NSFR.fc, self.device)
    #
    #     batch_size = self.args.batch_size_bs
    #     pos_img_num = self.pos_loader.dataset.__len__()
    #     neg_img_num = self.neg_loader.dataset.__len__()
    #
    #     # get predicates that need to be evaluated.
    #     pred_names = ['kp']
    #     for pi_c in pi_clauses:
    #         for body_atom in pi_c.body:
    #             if "inv_pred" in body_atom.pred.name:
    #                 pred_names.append(body_atom.pred.name)
    #
    #     C = len(pred_names)
    #     score_positive = torch.zeros((pos_img_num, C)).to(self.device)
    #     score_negative = torch.zeros((neg_img_num, C)).to(self.device)
    #
    #     for image_index in range(self.pos_loader.dataset.__len__()):
    #         V_T_list = NSFR.clause_eval_quick(pos_pred[image_index].unsqueeze(0)).detach()
    #         A = V_T_list.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
    #
    #         C_score = torch.zeros((C, batch_size)).to(self.device)
    #         # clause loop
    #         # for clause_index, V_T in enumerate(V_T_list):
    #         for pred_index, pred_name in enumerate(pred_names):
    #             predicted = NSFR.predict(v=V_T_list[0], predname=pred_name).detach()
    #             C_score[pred_index] = predicted
    #         # sum over positive prob
    #         score_positive[image_index, :] = C_score.squeeze(1)
    #
    #     # negative image loop
    #     for image_index in range(self.neg_loader.dataset.__len__()):
    #         V_T_list = NSFR.clause_eval_quick(neg_pred[image_index].unsqueeze(0)).detach()
    #         C_score = torch.zeros((C, batch_size)).to(self.device)
    #         for pred_index, pred_name in enumerate(pred_names):
    #             predicted = NSFR.predict(v=V_T_list[0], predname=pred_name).detach()
    #             C_score[pred_index] = predicted
    #             # C
    #             # C_score = PI.clause_eval(C_score)
    #             # sum over positive prob
    #         score_negative[image_index, :] = C_score.squeeze(1)
    #
    #     all_clause_scores = []
    #     for c_index in range(score_positive.shape[1]):
    #         clause_scores = [score_negative[:, c_index], score_positive[:, c_index]]
    #         all_clause_scores.append(clause_scores)
    #
    #     return all_clause_scores

    def eval_pi_language(self, bs_clauses, pi_languages, pos_pred, neg_pred):
        print("Eval PI Languages: ", len(pi_languages))
        pi_language_scores = torch.zeros(len(pi_languages))

        # scoring predicates
        for pi_index, pi_language in enumerate(pi_languages):

            lang, pi_clause = pi_language[0], pi_language[1]

            pred_names = [pi_clause[0].head.pred.name]
            # clauses = pi_clause
            atoms = logic_utils.get_atoms(lang)
            NSFR = get_nsfr_model(self.args, lang, pi_clause, atoms,
                                  self.NSFR.bk, self.bk_clauses, pi_clause, self.NSFR.fc, self.device)

            p_signs, p_goodness_scores, p_score_full_list = logic_utils.eval_predicates(NSFR, self.args, pred_names,
                                                                                        pos_pred, neg_pred)

            # = logic_utils.eval_predicates_sign(p_score)
            pi_language_scores[pi_index] = p_goodness_scores[0]

            # print(f"- Eval pi language {pi_index + 1}/{len(pi_languages)}")
            # for pi_c in pi_clause:
            #     print(pi_c)
            # print(f"- language score: {pi_language_scores[pi_index]}")
        pi_language_scores_sorted, pi_language_scores_sorted_indices = torch.sort(pi_language_scores, descending=True)
        passed_languages = []
        for index in pi_language_scores_sorted_indices:
            if pi_language_scores[index] == pos_pred.size(0):
                passed_languages.append(pi_languages[index])
            else:
                A = 12
                # print(f"insufficient language: {pi_languages[index][1]}")
                # print(f"insufficient score: {pi_language_scores[index]}")

        return passed_languages
        #     pred_type = None
        #     if p_goodness_scores[0] == p_goodness_scores[1]:
        #         # this is an output predicate
        #         pred_type = "output_predicate"
        #         output_pi_clauses.append(pi_clause)
        #         output_scores.append([pi_index] + p_goodness_scores)
        #         ip_names.append([pi_index, pred_names[1]])
        #     elif p_goodness_scores[0] <= p_goodness_scores[1]:
        #         # this is a hidden predicate
        #         pred_type = "hidden_predicate"
        #         hidden_pi_clauses.append(pi_clause)
        #         hidden_scores.append([pi_index] + p_goodness_scores)
        #         ip_names.append([pi_index, pred_names[1]])
        #     else:
        #         # this is not a good predicate
        #         pred_type = "archive_predicate"
        #         archive_pi_clauses.append(pi_clause)
        #         archive_scores.append([pi_index] + p_goodness_scores)
        #         ip_names.append([pi_index, pred_names[1]])
        #
        # # filter out predicates
        # output_ip = []
        # for output_score in output_scores:
        #     output_clause = output_pi_clauses[output_score[0]]
        #     ip_name = ip_names[output_score[0]]
        #     output_ip.append(ip_name)
        #     passed_pi_clauses_clusters.append(output_clause)
        #
        # hidden_ip = []
        # goodness_scores_sorted = sorted(hidden_scores, key=itemgetter(2), reverse=True)
        # goodness_scores_sorted_t5 = goodness_scores_sorted[:5]
        # for goodness_score in goodness_scores_sorted_t5:
        #     hidden_clause = hidden_pi_clauses[goodness_score[0]]
        #     ip_name = ip_names[goodness_score[0]]
        #     hidden_ip.append(ip_name)
        #     passed_pi_clauses_clusters.append(hidden_clause)
        #
        # archive_ip = []
        # for archive_score in archive_scores:
        #     archive_clause = archive_pi_clauses[archive_score[0]]
        #     unpassed_pi_clauses_clusters.append(archive_clause)
        #     ip_name = ip_names[archive_score[0]]
        #     archive_ip.append(ip_name)
        #     unpassed_pi_clauses_clusters.append(archive_clause)
        #
        # ip_indices = []
        # for ip_index, ip in enumerate(self.lang.invented_preds):
        #     if ip.name in hidden_ip:
        #         ip.ptype = "hidden_predicate"
        #         ip_indices.append(ip_index)
        #     elif ip.name in output_ip:
        #         ip.ptype = "output_predicate"
        #         ip_indices.append(ip_index)
        #     else:
        #         ip.ptype = "archive_predicate"
        #
        # hidden_predicates_indices = [ip[0] for ip in hidden_ip]
        # hidden_predicates = [self.lang.invented_preds[i] for i in hidden_predicates_indices]
        #
        # output_predicates_indices = [ip[0] for ip in output_ip]
        # output_predicates = [self.lang.invented_preds[i] for i in output_predicates_indices]
        #
        # self.lang.invented_preds = output_predicates + hidden_predicates
        #
        # passed_clauses = [c for c_cluster in passed_pi_clauses_clusters for c in c_cluster]
        # unpassed_clauses = [c for c_cluster in unpassed_pi_clauses_clusters for c in c_cluster]

    def extract_pi(self, passed_pi_languages):

        pi_clauses = []
        pi_predicates = []
        for index, passed_pi_language in enumerate(passed_pi_languages):
            passed_lang, passed_pi_clauses = passed_pi_language[0], passed_pi_language[1]
            for c in passed_pi_clauses:
                if c not in pi_clauses:
                    pi_clauses.append(c)
                else:
                    print(f"duplicate pi clause {c}")
            for p in passed_lang.invented_preds:
                if p not in pi_predicates:
                    pi_predicates.append(p)
        return pi_clauses, pi_predicates
