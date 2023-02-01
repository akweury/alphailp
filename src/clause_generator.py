from fol.logic import *
from nsfr_utils import update_nsfr_clauses, get_prob, get_nsfr_model
# from eval_clause import EvalInferModule
from refinement import RefinementGenerator
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pi_utils import get_pi_model


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

    def __init__(self, args, NSFR, PI, lang, pos_data_loader, neg_data_loader, mode_declarations, bk_clauses, device,
                 no_xil=False):
        self.args = args
        self.NSFR = NSFR
        self.PI = PI
        self.lang = lang
        self.mode_declarations = mode_declarations
        self.bk_clauses = bk_clauses
        self.device = device
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

    def generate(self, C_0, gen_mode='beam', T_beam=7, N_beam=20, N_max=100):
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
            beam_search_clauses = self.beam_search(C_0, T_beam=T_beam, N_beam=N_beam, N_max=N_max)
            return beam_search_clauses
        elif gen_mode == 'naive':
            return self.naive(C_0, T_beam=T_beam, N_max=N_max)

    def beam_search_clause(self, clause, T_beam=7, N_beam=20, N_max=100, th=0.98):
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
        B = [clause]
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
            loss_list = self.eval_clauses(refs)  # time-consuming line
            pi_loss_list = self.eval_pi_clauses(refs)

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

    def is_in_beam(self, B, clause, score):
        """If score is the same, same predicates => duplication
        """
        score = score.detach().cpu().numpy()
        preds = set([clause.head.pred] + [b.pred for b in clause.body])
        y = False
        for ci, score_i in B.items():
            score_i = score_i.detach().cpu().numpy()
            preds_i = set([clause.head.pred] + [b.pred for b in clause.body])
            if preds == preds_i and np.abs(score - score_i) < 1e-2:
                y = True
                # print("duplicated: ", clause, ci)
                break
        return y

    def beam_search(self, C_0, T_beam=7, N_beam=20, N_max=100):
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
        for clause in C_0:
            C = C.union(self.beam_search_clause(clause, T_beam, N_beam, N_max))
        C = sorted(list(C))
        print('======= BEAM SEARCHED CLAUSES ======')
        for c in C:
            print(c)
        return C

    def eval_pi_clauses(self, clauses):
        return None

    def eval_clauses(self, clauses):
        C = len(clauses)
        print("Eval clauses: ", len(clauses))
        # update infer module with new clauses
        # NSFR = update_nsfr_clauses(self.NSFR, clauses, self.bk_clauses, self.device)
        NSFR = get_nsfr_model(self.args, self.lang, clauses, self.NSFR.atoms, self.NSFR.bk, self.bk_clauses,
                              self.device)
        PI = get_pi_model(self.args, self.lang, clauses, self.NSFR.atoms, self.NSFR.bk, self.bk_clauses,
                          self.device)
        # TODO: Compute loss for validation data , score is bce loss
        # N C B G
        predicted_list_list = []

        score = torch.zeros((C,)).to(self.device)
        N_data = 0
        # List(C * B * G)

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
            V_T_list = NSFR.clause_eval(imgs).detach()
            C_score = torch.zeros((C, B)).to(self.device)
            for i, V_T in enumerate(V_T_list):
                # for each clause
                # B
                # print(V_T.shape)
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                # print("clause: ", clauses[i])
                # NSFR.print_valuation_batch(V_T)
                # print(predicted)
                # predicted = self.bce_loss(predicted, target_set)
                # predicted = torch.abs(predicted - target_set)
                # print(predicted)
                C_score[i] = predicted
            C_score = PI.clause_eval(C_score)
            # sum over positive prob
            C_score = C_score.sum(dim=1)
            score += C_score

        for i, sample in tqdm(enumerate(self.neg_loader, start=0)):
            imgs, target_set = map(lambda x: x.to(self.device), sample)
            # print(NSFR.clauses)
            img_array = imgs.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
            img_array_int8 = np.uint8(img_array * 255)
            img_pil = Image.fromarray(img_array_int8)
            # img_pil.show()
            N_data += imgs.size(0)
            B = imgs.size(0)
            # C * B * G
            V_T_list = NSFR.clause_eval(imgs).detach()
            C_score = torch.zeros((C, B)).to(self.device)
            for i, V_T in enumerate(V_T_list):
                # for each clause
                # B
                # print(V_T.shape)
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                # print("clause: ", clauses[i])
                # NSFR.print_valuation_batch(V_T)
                # print(predicted)
                # predicted = self.bce_loss(predicted, target_set)
                # predicted = torch.abs(predicted - target_set)
                # print(predicted)
                C_score[i] = 1 - predicted
            # C
            C_score = PI.clause_eval(C_score)
            # sum over positive prob
            C_score = C_score.sum(dim=1)
            score += C_score
        # return score
        # score = 1 - score.detach().cpu().numpy() / N_data
        return score


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

    def __init__(self, args, NSFR, PI, lang, pos_data_loader, neg_data_loader, mode_declarations, bk_clauses, device,
                 no_xil=False):
        self.args = args
        self.NSFR = NSFR
        self.PI = PI
        self.lang = lang
        self.mode_declarations = mode_declarations
        self.bk_clauses = bk_clauses
        self.device = device
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

    def generate(self, beam_search_clauses):
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
        pi_clauses = set()
        pi_clauses_candidates = list(beam_search_clauses)
        best_values = []
        best_clause_combinations = []
        value, _ = self.eval_multip_clauses(pi_clauses_candidates)  # time-consuming line
        best_values.append(value)
        for del_level in range(1, len(beam_search_clauses)):
            level_best_values = []
            level_best_combinations = []
            for i, pi_clauses_candidate in enumerate(pi_clauses_candidates):
                new_pi_clauses_candidate = pi_clauses_candidates[:i] + pi_clauses_candidates[i + 1:]
                new_value, best_del_index = self.eval_multip_clauses(new_pi_clauses_candidate)  # time-consuming line
                level_best_values.append(new_value)

                level_best_combinations.append(pi_clauses_candidates)

            print(f"========== level {del_level} ==================\n"
                  f"level values: {level_best_values}\n "
                  f"level best values: {np.max(level_best_values)}")


            level_best_index = np.argmax(level_best_values)
            pi_clauses_candidates.pop(level_best_index)

            best_values.append(np.max(level_best_values))
            best_clause_combinations.append(pi_clauses_candidates)
            print(f"left clauses: {pi_clauses_candidates}\n"
                  f"================================================")

        print(f"========best value in each level==========================\n"
              f"{best_values}")
        print(f"========best combinations in each level==========================\n"
              f"{best_clause_combinations}")
        # for i, ref in enumerate(refs):
        #     # check duplication
        #     if not self.is_in_beam(B_new, ref, loss_list[i]):
        #         B_new[ref] = loss_list[i]
        #         C_dic[ref] = loss_list[i]
        #
        #     # if len(C) >= N_max:
        #     #    break
        # B_new_sorted = sorted(B_new.items(), key=lambda x: x[1], reverse=True)
        # # top N_beam refiements
        # B_new_sorted = B_new_sorted[:N_beam]
        # # B_new_sorted = [x for x in B_new_sorted if x[1] > th]
        # for x in B_new_sorted:
        #     print(x[1], x[0])
        # B = [x[0] for x in B_new_sorted]
        # step += 1
        # if len(B) == 0:
        #     break

        best_index = np.argmax(best_values)
        pi_clauses = best_clause_combinations[best_index]
        pi_clauses_value =  np.max(best_values)
        print(f"best clause: {pi_clauses}")
        print(f"best clause value: {pi_clauses_value}")

        return pi_clauses

    def beam_search_clause(self, clause, T_beam=7, N_beam=20, N_max=100, th=0.98):
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
        B = [clause]
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
            loss_list = self.eval_multip_clauses(refs)  # time-consuming line
            pi_loss_list = self.eval_pi_clauses(refs)

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

    def is_in_beam(self, B, clause, score):
        """If score is the same, same predicates => duplication
        """
        score = score.detach().cpu().numpy()
        preds = set([clause.head.pred] + [b.pred for b in clause.body])
        y = False
        for ci, score_i in B.items():
            score_i = score_i.detach().cpu().numpy()
            preds_i = set([clause.head.pred] + [b.pred for b in clause.body])
            if preds == preds_i and np.abs(score - score_i) < 1e-2:
                y = True
                # print("duplicated: ", clause, ci)
                break
        return y

    def beam_search(self, C_0, T_beam=7, N_beam=20, N_max=100):
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
        for clause in C_0:
            C = C.union(self.beam_search_clause(clause, T_beam, N_beam, N_max))
        C = sorted(list(C))
        print('======= BEAM SEARCHED CLAUSES ======')
        for c in C:
            print(c)
        return C

    def eval_pi_clauses(self, clauses):
        return None

    def eval_multip_clauses(self, clauses):

        C = len(clauses)
        print("Eval clauses: ", len(clauses))
        # update infer module with new clauses
        # NSFR = update_nsfr_clauses(self.NSFR, clauses, self.bk_clauses, self.device)
        NSFR = get_nsfr_model(self.args, self.lang, clauses, self.NSFR.atoms, self.NSFR.bk, self.bk_clauses,
                              self.device)
        PI = get_pi_model(self.args, self.lang, clauses, self.NSFR.atoms, self.NSFR.bk, self.bk_clauses,
                          self.device)
        # TODO: Compute loss for validation data , score is bce loss
        # N C B G
        predicted_list_list = []

        score = torch.zeros((C,)).to(self.device)
        N_data = 0
        # List(C * B * G)

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
            V_T_list = NSFR.clause_eval(imgs).detach()
            C_score = torch.zeros((C, B)).to(self.device)
            for i, V_T in enumerate(V_T_list):
                # for each clause
                # B
                # print(V_T.shape)
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                # print("clause: ", clauses[i])
                # NSFR.print_valuation_batch(V_T)
                # print(predicted)
                # predicted = self.bce_loss(predicted, target_set)
                # predicted = torch.abs(predicted - target_set)
                # print(predicted)
                C_score[i] = predicted
            # C_score = PI.clause_eval(C_score)
            C_score = C_score.max()
            # sum over positive prob

            score += C_score

        for i, sample in tqdm(enumerate(self.neg_loader, start=0)):
            imgs, target_set = map(lambda x: x.to(self.device), sample)
            # print(NSFR.clauses)
            img_array = imgs.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
            img_array_int8 = np.uint8(img_array * 255)
            img_pil = Image.fromarray(img_array_int8)
            # img_pil.show()
            N_data += imgs.size(0)
            B = imgs.size(0)
            # C * B * G
            V_T_list = NSFR.clause_eval(imgs).detach()
            C_score = torch.zeros((C, B)).to(self.device)
            for i, V_T in enumerate(V_T_list):
                # for each clause
                # B
                # print(V_T.shape)
                predicted = NSFR.predict(v=V_T, predname='kp').detach()
                # print("clause: ", clauses[i])
                # NSFR.print_valuation_batch(V_T)
                # print(predicted)
                # predicted = self.bce_loss(predicted, target_set)
                # predicted = torch.abs(predicted - target_set)
                # print(predicted)
                C_score[i] = 1 - predicted
            # C
            # C_score = PI.clause_eval(C_score)
            # sum over positive prob
            C_score = C_score.max()
            score += C_score
        # return score
        # score = 1 - score.detach().cpu().numpy() / N_data
        best_score = score.max()
        best_score_index = np.argmax(best_score.to("cpu"))
        return best_score.to("cpu"), best_score_index
