from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

import chart_utils
import eval_clause_infer
from refinement import RefinementGenerator
from percept import YOLOPerceptionModule
from nsfr_utils import update_nsfr_clauses, get_prob, get_nsfr_model
import logic_utils
from fol.language import DataType
import log_utils
import eval_utils
from fol.data_utils import DataUtils
import eval_clause_infer
import config


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

    def __init__(self, args, NSFR, PI, lang, pos_data_loader, neg_data_loader, mode_declarations,
                 no_xil=False):
        self.args = args
        self.NSFR = NSFR
        self.PI = PI
        self.lang = lang
        self.mode_declarations = mode_declarations
        self.bk_clauses = None
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

    def extend_clauses(self, clauses, args, pi_clauses):
        refs = []
        B_ = []
        is_done = False
        for c in clauses:
            refs_i = self.rgen.refinement_clause(c)
            unused_args, used_args = log_utils.get_unused_args(c)
            refs_i_removed = logic_utils.remove_duplicate_clauses(refs_i, unused_args, used_args, args)
            # remove invalid clauses
            ###refs_i = [x for x in refs_i if self._is_valid(x)]
            # remove already appeared refs
            refs_i_removed = list(set(refs_i_removed).difference(set(B_)))
            B_.extend(refs_i_removed)
            refs.extend(refs_i_removed)

        # remove semantic conflict clauses
        refs_no_conflict = self.remove_conflict_clauses(refs, pi_clauses, args)
        if len(refs_no_conflict) == 0:
            is_done = True
        return refs_no_conflict, is_done

    def clause_extension(self, init_clauses, pos_pred, neg_pred, pi_clauses, args, max_clause,
                         max_step=4, iteration=None, max_iteration=None, no_new_preds=False, last_refs=[]):
        index_pos = config.score_example_index["pos"]
        index_neg = config.score_example_index["neg"]

        log_utils.add_lines(f"\n=== beam search iteration {iteration}/{max_iteration} ===", args.log_file)
        eval_pred = ['kp']
        clause_with_scores = []
        # extend clauses
        step = 0
        is_done = False
        refs = init_clauses
        if no_new_preds:
            step = max_step
            refs = last_refs
        if args.pi_top == 0:
            step = max_step
            if len(last_refs) > 0:
                refs = last_refs
        while step <= max_step:
            # log
            log_utils.print_time(args, iteration, step, max_step)
            # clause extension
            refs_extended, is_done = self.extend_clauses(refs, args, pi_clauses)
            # update NSFR
            self.NSFR = get_nsfr_model(args, self.lang, refs_extended, self.NSFR.atoms, pi_clauses, self.NSFR.fc)
            # evaluate new clauses
            score_all = eval_clause_infer.eval_clause_on_scenes(self.NSFR, args, eval_pred, pos_pred, neg_pred)
            scores = eval_clause_infer.eval_clauses(score_all[:, :, index_pos], score_all[:, :, index_neg], args)
            # classify clauses
            clause_with_scores = eval_clause_infer.classify_clauses(refs_extended, score_all, scores)
            # print best clauses that have been found...
            clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)

            new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
            max_clause, found_sn = self.check_result(clause_with_scores, higher, max_clause, new_max)

            if args.pi_top > 0:
                refs, clause_with_scores, is_done = self.prune_clauses(clause_with_scores, args)
            else:
                refs = logic_utils.top_select(clause_with_scores, args)
            step += 1

            if found_sn or len(refs) == 0:
                is_done = True
                break

        return clause_with_scores, max_clause, step, refs, is_done

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

    def is_in_beam(self, B, clause):
        """If score is the same, same predicates => duplication
        """
        # TODO: simplify this segment.
        clause_preds = set([clause.head.pred] + [b.pred for b in clause.body])
        clause_body_sorted = sorted(clause.body)
        clause_terms = clause.head.terms + [t for b in clause_body_sorted for t in b.terms]
        y = False
        for beam_clause in B:
            bs_clause_pred = set([beam_clause.head.pred] + [b.pred for b in beam_clause.body])
            bs_body_sorted = sorted(beam_clause.body)
            bs_clause_terms = beam_clause.head.terms + [t for b in bs_body_sorted for t in b.terms]
            if clause_preds == bs_clause_pred and clause_terms == bs_clause_terms:
                y = True
                # print("duplicated: ", clause, ci)
                break
        return y

    # def eval_pi_clauses(self, clauses):
    #     return None

    # def eval_clause_sign(self, p_scores):
    #     resolution = 2
    #
    #     p_clauses_signs = []
    #     for p_score in p_scores.values():
    #         clause_sign_list = []
    #         clause_score_list = []
    #         clause_score_full_list = []
    #         for clause_image_score in p_score:
    #             data_map = np.zeros(shape=[resolution, resolution])
    #             for index in range(len(clause_image_score[0][0])):
    #                 x_index = int(clause_image_score[0][0, index] * resolution)
    #                 y_index = int(clause_image_score[1][0, index] * resolution)
    #                 data_map[x_index, y_index] += 1
    #
    #             pos_low_neg_low_area = data_map[0, 0]
    #             pos_high_neg_low_area = data_map[0, 1]
    #             pos_low_neg_high_area = data_map[1, 0]
    #             pos_high_neg_high_area = data_map[1, 1]
    #
    #             # TODO: find a better score evaluation function
    #             clause_score = pos_high_neg_low_area - pos_high_neg_high_area
    #             clause_score_list.append(clause_score)
    #             clause_score_full_list.append(
    #                 [pos_low_neg_low_area, pos_high_neg_low_area, pos_low_neg_high_area, pos_high_neg_high_area])
    #
    #             data_map[0, 0] = 0
    #             if np.max(data_map) == data_map[0, 1] and data_map[0, 1] > data_map[1, 1]:
    #                 clause_sign_list.append(True)
    #             else:
    #                 clause_sign_list.append(False)
    #         p_clauses_signs.append([clause_sign_list, clause_score_list, clause_score_full_list])
    #     return p_clauses_signs

    # def eval_clauses(self, clauses, pos_pm_res, neg_pm_res):
    #     C = len(clauses)
    #     print("Eval clauses: ", len(clauses))
    #     # update infer module with new clauses
    #     # NSFR = update_nsfr_clauses(self.NSFR, clauses, self.bk_clauses, self.device)
    #     pi_clauses = []
    #     batch_size = self.args.batch_size_bs
    #     NSFR = get_nsfr_model(self.args, self.lang, clauses, self.NSFR.atoms,
    #                           self.NSFR.bk, self.bk_clauses, pi_clauses, self.NSFR.fc, self.device)
    #     # TODO: Compute loss for validation data , score is bce loss
    #     pos_img_num = self.pos_loader.dataset.__len__()
    #     neg_img_num = self.neg_loader.dataset.__len__()
    #
    #     positive_score = torch.zeros((pos_img_num, C)).to(self.device)
    #     negative_score = torch.zeros((neg_img_num, C)).to(self.device)
    #
    #     for i in range(self.pos_loader.dataset.__len__()):
    #         V_T_list = NSFR.clause_eval_quick(pos_pm_res[i].unsqueeze(0)).detach()
    #         C_score = torch.zeros((C, batch_size)).to(self.device)
    #         for j, V_T in enumerate(V_T_list):
    #             predicted = NSFR.ilp_predict(v=V_T, predname='kp').detach()
    #             C_score[j] = predicted
    #         # C_score = PI.clause_eval(C_score)
    #         # sum over positive prob
    #         C_score = C_score.sum(dim=1)
    #         positive_score[i, :] = C_score
    #     a = positive_score.detach().numpy()
    #
    #     for i in range(self.neg_loader.dataset.__len__()):
    #         V_T_list = NSFR.clause_eval_quick(neg_pm_res[i].unsqueeze(0)).detach()
    #         C_score = torch.zeros((C, batch_size)).to(self.device)
    #         for j, V_T in enumerate(V_T_list):
    #             predicted = NSFR.ilp_predict(v=V_T, predname='kp').detach()
    #             C_score[j] = predicted
    #         # C_score = PI.clause_eval(C_score)
    #         # sum over positive prob
    #         # C_score = PI.clause_eval(C_score)
    #         # sum over positive prob
    #         C_score = C_score.sum(dim=1)
    #         negative_score[i] = C_score
    #     b = negative_score.detach().numpy()
    #
    #     # positive_clauses = []
    #     all_clauses_scores = []
    #     for c_index in range(positive_score.shape[1]):
    #         clause_scores = [negative_score[:, c_index], positive_score[:, c_index]]
    #         clause_sign, clause_score, clause_score_full = self.eval_clause_sign(clause_scores)
    #         all_clauses_scores.append(clause_score)
    #         # if clause_sign:
    #         #     positive_clauses.append(clauses[c_index])
    #         # plot the clause evaluation
    #
    #         # clause_scores_reverse = [positive_score[:, c_index], negative_score[:, c_index]]
    #         # chart_utils.plot_scatter_chart([clause_scores_reverse], config.buffer_path / "img",
    #         #                                f"scatter_ce_all_{len(clauses)}_{c_index}",
    #         #                                labels=f"{str(clauses[c_index]) + str(clause_sign)}",
    #         #                                x_label="positive score", y_label="negative score")
    #
    #     return all_clauses_scores

    def remove_conflict_clauses(self, refs, pi_clauses, args):
        # remove conflict clauses
        refs_non_conflict = logic_utils.remove_conflict_clauses(refs, pi_clauses, args)
        refs_non_trivial = logic_utils.remove_trivial_clauses(refs_non_conflict, args)

        log_utils.add_lines(f"after removing conflict clauses: {len(refs_non_trivial)} clauses left", args.log_file)
        return refs_non_trivial

    def print_clauses(self, clause_dict, args):
        log_utils.add_lines('\n======= BEAM SEARCHED CLAUSES ======', args.log_file)

        if len(clause_dict["sn"]) > 0:
            for c in clause_dict["sn"]:
                log_utils.add_lines(f"sufficient and necessary clause: {c[0]}", args.log_file)
        if len(clause_dict["sn_good"]) > 0:
            for c in clause_dict["sn_good"]:
                score = logic_utils.get_four_scores(c[1].unsqueeze(0))
                log_utils.add_lines(
                    f"sufficient and necessary clause with {args.sn_th * 100}% accuracy: {c[0]}, {score}",
                    args.log_file)
        if len(clause_dict["sc"]) > 0:
            for c in clause_dict["sc"]:
                score = logic_utils.get_four_scores(c[1].unsqueeze(0))
                log_utils.add_lines(f"sufficient clause: {c[0]}, {score}", args.log_file)
        if len(clause_dict["sc_good"]) > 0:
            for c in clause_dict["sc_good"]:
                score = logic_utils.get_four_scores(c[1].unsqueeze(0))
                log_utils.add_lines(f"sufficient clause with {args.sc_th * 100}%: {c[0]}, {score}", args.log_file)
        if len(clause_dict["nc"]) > 0:
            for c in clause_dict["nc"]:
                score = logic_utils.get_four_scores(c[1].unsqueeze(0))
                log_utils.add_lines(f"necessary clause: {c[0]}, {score}", args.log_file)
        if len(clause_dict["nc_good"]) > 0:
            for c in clause_dict["nc_good"]:
                score = logic_utils.get_four_scores(c[1].unsqueeze(0))
                log_utils.add_lines(f"necessary clause with {args.nc_th * 100}%: {c[0]}, {score}", args.log_file)
        log_utils.add_lines('============= Beam search End ===================\n', args.log_file)

    def update_refs(self, clause_with_scores, args):
        refs = []
        nc_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_with_scores, "clause", args)
        refs += nc_clauses
        # if priority == "nc":
        #     nc_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_dict['nc'], "nc", args)
        #     refs += nc_clauses
        #
        # if priority == "nc_good":
        #     nc_good_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_dict['nc_good'], "nc_good", args)
        #     refs += nc_good_clauses
        #
        # if priority == "sc":
        #     sc_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_dict['sc'], "sc", args)
        #     refs += sc_clauses
        #
        # if priority == "sc_good":
        #     sc_good_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_dict['sc_good'], "sc_good", args)
        #     refs += sc_good_clauses
        #
        # if priority == "uc_good":
        #     uc_good_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_dict['uc_good'], "uc_good", args)
        #     refs += uc_good_clauses
        #
        # if priority == "uc":
        #     uc_clauses = logic_utils.extract_clauses_from_bs_clauses(clause_dict['uc'], "uc", args)
        #     refs += uc_clauses

        return refs

    # def select_all_refs(self, clause_dict,args):
    #     refs = []
    #     refs += logic_utils.extract_clauses_from_bs_clauses(clause_dict['nc'],args)
    #     refs += logic_utils.extract_clauses_from_bs_clauses(clause_dict['sc'],args)
    #     refs += logic_utils.extract_clauses_from_bs_clauses(clause_dict['uc'],args)
    #     return refs

    def is_in_beam_same_score(self, B, clause, c_i, scores):

        """If score is the same, same predicates => duplication
        """
        # TODO: simplify this segment.
        clause_preds = set([clause.head.pred] + [b.pred for b in clause.body])
        # clause_body_sorted = sorted(clause.body)
        # clause_terms = clause.head.terms + [t for b in clause_body_sorted for t in b.terms]
        y = False
        for bc_i, beam_clause in enumerate(B):
            bs_clause_pred = set([beam_clause.head.pred] + [b.pred for b in beam_clause.body])
            # bs_body_sorted = sorted(beam_clause.body)
            # bs_clause_terms = beam_clause.head.terms + [t for b in bs_body_sorted for t in b.terms]
            if clause_preds == bs_clause_pred and torch.equal(scores[c_i], scores[bc_i]):
                y = True
                # print("duplicated: ", clause, ci)
                break
        return y

    def prune_clauses(self, clause_with_scores, args):
        refs = []

        # prune score similar clauses
        log_utils.add_lines(f"=============== score pruning ==========", args.log_file)

        if args.score_unique:
            score_unique_c = []
            appeared_scores = []
            for c in clause_with_scores:
                if not eval_clause_infer.eval_score_similarity(c[1][2], appeared_scores, args.similar_th):
                    score_unique_c.append(c)
                    appeared_scores.append(c[1][2])
            c_score_pruned = score_unique_c
        else:
            c_score_pruned = clause_with_scores

        # prune predicate similar clauses
        log_utils.add_lines(f"=============== semantic pruning ==========", args.log_file)
        if args.semantic_unique:
            semantic_unique_c = []
            semantic_repeat_c = []
            appeared_semantics = []
            for c in c_score_pruned:
                c_semantic = logic_utils.get_semantic_from_c(c[0])
                if not eval_clause_infer.eval_semantic_similarity(c_semantic, appeared_semantics, args):
                    semantic_unique_c.append(c)
                    appeared_semantics.append(c_semantic)
                else:
                    semantic_repeat_c.append(c)
            c_semantic_pruned = semantic_unique_c
            for c in c_semantic_pruned:
                log_utils.add_lines(f"(unique semantic clause) {c[0]}", args.log_file)
            for c in semantic_repeat_c:
                log_utils.add_lines(f"(repeat semantic clause) {c[0]}", args.log_file)
        else:
            c_semantic_pruned = c_score_pruned

        c_score_pruned = c_semantic_pruned
        # select top N clauses
        if args.c_top is not None and len(c_score_pruned) > args.c_top:
            c_score_pruned = c_score_pruned[:args.c_top]
        log_utils.add_lines(f"after top select: {len(c_score_pruned)}", args.log_file)

        refs += self.update_refs(c_score_pruned, args)

        return refs, c_score_pruned, False

    def check_result(self, clause_with_scores, higher, max_clause, new_max_clause):

        if higher:
            best_clause = new_max_clause
        else:
            best_clause = max_clause

        if clause_with_scores[0][1][2] == 1.0:
            return best_clause, True
        elif clause_with_scores[0][1][2] > self.args.sn_th:
            return best_clause, True
        return best_clause, False



def count_arity_from_clause_cluster(clause_cluster):
    arity_list = []
    for [c_i, clause, c_score] in clause_cluster:
        for b in clause.body:
            if "in" == b.pred.name:
                continue
            for t in b.terms:
                if t.name not in arity_list and "O" in t.name:
                    arity_list.append(t.name)
    # arity = len(arity_list)
    arity_list.sort()
    return arity_list


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

    def __init__(self, args, NSFR, PI, lang, pos_data_loader, neg_data_loader, mode_declarations,
                 no_xil=False):
        self.args = args
        self.lang = lang
        self.NSFR = NSFR
        self.PI = PI
        self.bk_clauses = None
        self.device = args.device
        self.pos_loader = pos_data_loader
        self.neg_loader = neg_data_loader

    def generate(self, beam_search_clauses, pi_clauses, pos_pred, neg_pred, args, step):
        found_ns = False
        # evaluate for all the clauses
        # clause_image_scores = self.eval_multi_clauses(beam_search_clauses, pos_pred, neg_pred)  # time-consuming line
        # clause_signs, clause_scores, clause_scores_full = self.eval_clause_sign(clause_image_scores)

        # clause_candidates = logic_utils.eval_clause_clusters(clause_clusters, p_scores_list)

        # generate new clauses
        # sc_new_predicates = []
        # sc_good_new_predicates = []
        # nc_new_predicates = []
        # nc_good_new_predicates = []
        # uc_new_predicates = []
        # uc_good_new_predicates = []
        # nc_sc_new_predicates = []

        new_predicates, found_ns = self.cluster_invention(beam_search_clauses, pi_clauses, pos_pred.shape[0], args)
        log_utils.add_lines(f"new PI: {len(new_predicates)}\n", args.log_file)
        for new_c, new_c_score in new_predicates:
            log_utils.add_lines(f"{new_c} {new_c_score.reshape(3)}", args.log_file)

        # cluster sufficient clauses
        # if len(beam_search_clauses['sc']) > 1:
        #     sc_new_predicates, found_ns = self.cluster_invention(beam_search_clauses["sc"], pi_clauses,
        #                                                          pos_pred.shape[0], args)
        #     log_utils.add_lines(f"new PI from sc: {len(sc_new_predicates)}\n", args.log_file)
        # if len(beam_search_clauses['sc_good']) > 0:
        #     sc_good_new_predicates, found_ns = self.cluster_invention(beam_search_clauses["sc_good"], pi_clauses,
        #                                                               pos_pred.shape[0], args)
        #
        #     log_utils.add_lines(f"new PI from sc_good: {len(sc_good_new_predicates)}", args.log_file)
        #     # for p in sc_good_new_predicates:
        #     #     print(p)
        #
        # if not found_ns and len(beam_search_clauses['nc']) > 0:
        #     nc_new_predicates, found_ns = self.cluster_invention(beam_search_clauses["nc"], pi_clauses,
        #                                                          pos_pred.shape[0], args, random_top=args.nc_good_top)
        #     log_utils.add_lines(f"\nnew PI from nc: {len(nc_new_predicates)}", args.log_file)
        #     # for p in nc_new_predicates:
        #     #     print(p)
        #
        # if not found_ns and len(beam_search_clauses['sc']) > 0 and len(beam_search_clauses['nc']) > 0:
        #     nc_sc_new_predicates, found_ns = self.cluster_invention(beam_search_clauses["sc"], pi_clauses,
        #                                                             pos_pred.shape[0], args)
        #     log_utils.add_lines(f"\nnew PI from nc+sc: {len(nc_sc_new_predicates)}", args.log_file)
        #     # for p in nc_sc_new_predicates:
        #     #     print(p)
        #
        # if not found_ns and len(beam_search_clauses['nc_good']) > 0:
        #     nc_good_new_predicates, found_ns = self.cluster_invention(beam_search_clauses["nc_good"], pi_clauses,
        #                                                               pos_pred.shape[0], args)
        #     log_utils.add_lines(f"\nnew PI from nc_good: {len(nc_good_new_predicates)}", args.log_file)
        #     # for p in nc_good_new_predicates:
        #     #     print(p)
        # # # cluster necessary clauses
        # if not found_ns and len(beam_search_clauses['uc_good']) > 0:
        #     uc_good_new_predicates, found_ns = self.cluster_invention(beam_search_clauses["uc_good"], pi_clauses,
        #                                                               pos_pred.shape[0], args)
        #     log_utils.add_lines(f"\nnew PI from UC_GOOD: {len(uc_good_new_predicates)}", args.log_file)
        #     # for p in uc_good_new_predicates:
        #     #     print(p)
        # if not found_ns:
        #     uc_new_predicates, found_ns = self.cluster_invention(beam_search_clauses["uc"], pi_clauses,
        #                                                          pos_pred.shape[0], args, random_top=args.uc_top)
        #     log_utils.add_lines(f"\nnew PI from UC: {len(uc_new_predicates)}", args.log_file)
        #     # for p in uc_new_predicates:
        #     #     print(p)

        # top_selector = args.pi_top
        # sc_new_predicates = self.prune_predicates(sc_new_predicates, args, keep_all=True)[:top_selector]
        # sc_good_new_predicates = self.prune_predicates(sc_good_new_predicates, args, keep_all=True)[:top_selector]
        # nc_new_predicates = self.prune_predicates(nc_new_predicates, args)[:top_selector]
        # nc_good_new_predicates = self.prune_predicates(nc_good_new_predicates, args)[:top_selector]
        # uc_good_new_predicates = self.prune_predicates(uc_good_new_predicates, args)[:top_selector]
        # uc_new_predicates = self.prune_predicates(uc_new_predicates, args)[:top_selector]
        # nc_sc_new_predicates = self.prune_predicates(nc_sc_new_predicates, args)[:top_selector]
        # new_predicates = sc_new_predicates + uc_new_predicates + nc_new_predicates + sc_good_new_predicates + \
        #                  nc_good_new_predicates + uc_good_new_predicates + nc_sc_new_predicates
        # convert to strings
        new_clauses_str_list, kp_str_list = self.generate_new_clauses_str_list(new_predicates)

        # convert clauses from strings to objects
        # pi_languages = logic_utils.get_pi_clauses_objs(self.args, self.lang, new_clauses_str_list, new_predicates)
        du = DataUtils(lark_path=args.lark_path, lang_base_path=args.lang_base_path, dataset_type=args.dataset_type,
                       dataset=args.dataset)
        lang, init_clauses, bk_pi_clauses, atoms = logic_utils.get_lang(args)
        for learned_p in self.lang.invented_preds:
            lang.invented_preds.append(learned_p)

        all_pi_clauses, all_pi_kp_clauses = du.gen_pi_clauses(lang, new_predicates, new_clauses_str_list, kp_str_list)

        # pos_pred = pos_pred.to(self.args.device)
        # neg_pred = neg_pred.to(self.args.device)
        # generate pi clauses
        # passed_pi_languages = self.eval_pi_language(beam_search_clauses, pi_languages, pos_pred, neg_pred)
        # # passed_pi_languages = passed_pi_languages[:5]
        #
        all_pi_clauses = self.extract_pi(lang, all_pi_clauses, args) + pi_clauses
        all_pi_kp_clauses = self.extract_kp_pi(lang, all_pi_kp_clauses, args) + pi_clauses

        log_utils.add_lines(f"======  Total PI Number: {len(self.lang.invented_preds)}  ======", args.log_file)
        for p in self.lang.invented_preds:
            log_utils.add_lines(f"{p}", args.log_file)
        log_utils.add_lines(f"========== Total {len(all_pi_clauses)} PI Clauses ============= ", args.log_file)
        for c in all_pi_clauses:
            log_utils.add_lines(f"{c}", args.log_file)

        return all_pi_clauses, all_pi_kp_clauses, found_ns

    def eval_multi_clauses(self, clauses, pos_pred, neg_pred, args):

        C = len(clauses)
        log_utils.add_lines(f"Eval clauses: {len(clauses)}", args.log_file)
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
                predicted = NSFR.ilp_predict(v=V_T, predname='kp').detach()
                C_score[clause_index] = predicted
            # sum over positive prob
            score_positive[image_index, :] = C_score.squeeze(1)

        # negative image loop
        for image_index in range(self.neg_loader.dataset.__len__()):
            V_T_list = NSFR.clause_eval_quick(neg_pred[image_index].unsqueeze(0)).detach()
            C_score = torch.zeros((C, batch_size)).to(self.device)
            for clause_index, V_T in enumerate(V_T_list):
                predicted = NSFR.ilp_predict(v=V_T, predname='kp').detach()
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

    def generate_new_predicate(self, clause_clusters, clause_type=None):
        new_predicate = None
        # positive_clauses_exchange = [(c[1], c[0]) for c in positive_clauses]
        # no_hn_ = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0 and c[0][3] == 0]
        # no_hnlp = [(c[0], c[1]) for c in positive_clauses_exchange if c[0][2] == 0]
        # score clauses properly

        new_predicates = []
        # cluster predicates
        for pi_index, [clause_cluster, cluster_score] in enumerate(clause_clusters):
            args = count_arity_from_clause_cluster(clause_cluster)
            dtypes = [DataType("object")] * len(args)
            new_predicate = self.lang.get_new_invented_predicate(arity=len(args), pi_dtypes=dtypes, args=args,
                                                                 pi_types=clause_type)
            new_predicate.body = []
            for [c_i, clause, c_score] in clause_cluster:
                atoms = []
                for atom in clause.body:
                    terms = logic_utils.get_terms_from_atom(atom)
                    terms = sorted(terms)
                    if "X" in terms:
                        terms.remove("X")
                    obsolete_term = [t for t in terms if t not in args]
                    if len(obsolete_term) == 0:
                        atoms.append(atom)
                new_predicate.body.append(atoms)
            if len(new_predicate.body) > 1:
                new_predicates.append([new_predicate, cluster_score])
            elif len(new_predicate.body) == 1:
                body = (new_predicate.body)[0]
                if len(body) > new_predicate.arity + 1:
                    new_predicates.append([new_predicate, cluster_score])
            # # symmetric invent
            # new_sym_predicate = self.lang.get_new_invented_predicate(arity=arity, pi_dtypes=dtypes)
            # symmetry_bodies = []
            # for b_list in new_predicate.body:
            #     symmetry_b = []
            #     for b in b_list:
            #         if "at_area" in b.pred.name:
            #             sym_b_pred = logic.Predicate(b.pred.name, arity, dtypes)
            #             sym_b_terms = (logic.Var(b.terms[1].name), logic.Var(b.terms[0].name))
            #             sym_b = logic.Atom(sym_b_pred, sym_b_terms)
            #             symmetry_b.append(sym_b)
            #         else:
            #             symmetry_b.append(b)
            #     symmetry_bodies.append(symmetry_b)
            # new_sym_predicate.body = symmetry_bodies + new_predicate.body
            # new_predicates.append(new_sym_predicate)

        return new_predicates

    def generate_new_clauses_str_list(self, new_predicates):
        pi_str_lists = []
        kp_str_lists = []
        for [new_predicate, p_score] in new_predicates:
            single_pi_str_list = []
            # head_args = "(O1,O2)" if new_predicate.arity == 2 else "(X)"
            kp_clause = "kp(X):-"
            head_args = "("

            for arg in new_predicate.args:
                head_args += arg + ","
                kp_clause += f"in({arg},X),"
            head_args = head_args[:-1]
            head_args += ")"
            kp_clause += f"{new_predicate.name}{head_args}."
            kp_str_lists.append(kp_clause)

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
            pi_str_lists.append([single_pi_str_list, p_score])
        # for p_i, p_list in enumerate(pi_str_lists):
        # for p in p_list:
        #     print(f"{p_i}/{len(pi_str_lists)} Invented Predicate: {p}")
        return pi_str_lists, kp_str_lists

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

            scores = eval_clause_infer.eval_clause_on_scenes(NSFR, self.args, pred_names, pos_pred, neg_pred)

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
                # print(f"unnecessary language: {pi_languages[index][1]}")
                # print(f"unnecessary score: {pi_language_scores[index]}")

        return passed_languages

    def extract_pi(self, new_lang, all_pi_clauses, args):
        for index, new_p in enumerate(new_lang.invented_preds):
            if new_p in self.lang.invented_preds:
                continue
            is_duplicate = False
            for self_p in self.lang.invented_preds:
                if new_p.body == self_p.body:
                    is_duplicate = True
                    log_utils.add_lines(f"duplicate pi body {new_p.name} {new_p.body}", args.log_file)
                    break
            if not is_duplicate:
                print(f"add new predicate: {new_p.name}")
                self.lang.invented_preds.append(new_p)
            else:
                log_utils.add_lines(f"duplicate pi: {new_p}", args.log_file)

        new_p_names = [self_p.name for self_p in self.lang.invented_preds]
        new_all_pi_clausese = []
        for pi_c in all_pi_clauses:
            pi_c_head_name = pi_c.head.pred.name
            if pi_c_head_name in new_p_names:
                new_all_pi_clausese.append(pi_c)
        return new_all_pi_clausese

    def extract_kp_pi(self, new_lang, all_pi_clauses, args):
        new_all_pi_clausese = []
        for pi_c in all_pi_clauses:
            pi_c_head_name = pi_c.head.pred.name
            new_all_pi_clausese.append(pi_c)
        return new_all_pi_clausese

    def cluster_invention(self, clause_candidates, pi_clauses, total_score, args, random_top=None, searching_for=None):
        found_ns = False

        clu_lists = logic_utils.search_independent_clauses_parallel(clause_candidates, total_score, args)
        new_predicates = self.generate_new_predicate(clu_lists)
        new_predicates = new_predicates[:args.pi_top]
        # if len(sn_clu) > 0:
        #     found_ns = True
        #     new_predicates = self.generate_new_predicate(sn_clu)
        # if len(sn_th_clu) > 0:
        #     new_predicates += self.generate_new_predicate(sn_th_clu)
        # if len(n_clu) > 0:
        #     new_predicates += self.generate_new_predicate(n_clu)
        # # if len(nc_th_clu) > 0:
        # #     new_predicates += self.generate_new_predicate(nc_th_clu)[:5]
        # if len(s_clu) > 0:
        #     new_predicates += self.generate_new_predicate(s_clu)
        # if len(sc_th_clu) > 0:
        #     new_predicates += self.generate_new_predicate(sc_th_clu)[:10]

        return new_predicates, found_ns

    def prune_predicates(self, new_predicates, args, keep_all=False):

        no_duplicate = logic_utils.remove_duplicate_predicates(new_predicates, args)
        if len(no_duplicate) > 0:
            new_predicates = no_duplicate

        no_same_four = logic_utils.remove_same_four_score_predicates(new_predicates, args)
        return new_predicates
