import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_utils import softor, weight_sum


def init_identity_weights(X, device):
    ones = torch.ones((X.size(0),), dtype=torch.float32) * 100
    return torch.diag(ones).to(device)


class InferModule(nn.Module):
    def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk=None, I_pi=None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(InferModule, self).__init__()
        self.I = I
        self.I_bk = I_bk
        self.I_pi = I_pi
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        self.beta = 0.1  # softmax temperature
        if not train:
            self.W = init_identity_weights(I, device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.tensor(
                # np.random.normal(size=(m, I.size(0))), requires_grad=True, dtype=torch.float32).to(device))
                np.random.rand(m, I.size(0)), requires_grad=True, dtype=torch.float32).to(device))
        # clause functions
        self.cs = [ClauseFunction(I[i], gamma=gamma)
                   for i in range(self.I.size(0))]

        if not I_bk is None:
            self.cs_bk = [ClauseFunction(I_bk[i], gamma=gamma) for i in range(self.I_bk.size(0))]
            self.W_bk = init_identity_weights(I_bk, device)
        if I_pi is not None:
            self.cs_pi = [ClauseFunction(I_pi[i], gamma=gamma) for i in range(self.I_pi.size(0))]
            self.W_pi = init_identity_weights(I_pi, device)

        # print("W: ", self.W.shape)
        # print("W_bk: ", self.W_bk)

        # assert m == self.C, "Invalid m and C: " + \
        #    str(m) + ' and ' + str(self.C)

    def get_params(self):
        assert self.train_, "Infer module is not in training mode."
        return [self.W]

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        R = x
        if self.I_bk is None:
            for t in range(self.infer_step):
                R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        else:
            for t in range(self.infer_step):
                # R = softor([R, self.r_bk(R)], dim=1, gamma=self.gamma)
                # a = R.detach().to("cpu").numpy().reshape(-1, 1)

                # tic = time.perf_counter()

                r_R = self.r(R)
                # b = r_R.detach().to("cpu").numpy().reshape(-1, 1)

                # toc = time.perf_counter()

                r_bk_R = self.r_bk(R)
                # c = r_bk_R.detach().to("cpu").numpy().reshape(-1, 1)

                # toc_2 = time.perf_counter()

                if self.I_pi is not None:
                    r_pi_R = self.r_pi(R)
                    # d = r_pi_R.detach().to("cpu").numpy().reshape(-1, 1)
                    # toc_3 = time.perf_counter()

                    R = softor([R, r_R, r_bk_R, r_pi_R], dim=1, gamma=self.gamma)
                else:
                    R = softor([R, r_R, r_bk_R], dim=1, gamma=self.gamma)

                # print(f"Calculate r_R in {toc - tic:0.4f} seconds")
                # print(f"Calculate r_bk_R in {toc_2 - toc:0.4f} seconds")
                # print(f"Calculate r_pi_R in {toc_3 - toc_2:0.4f} seconds")

        # z = R.detach().to("cpu").numpy().reshape(-1, 1)
        return R

    def r(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C

        clause_function_value_lists = []
        for i in range(self.I.size(0)):
            clause_function = self.cs[i]
            # TODO: Think about the clause function of pi clauses
            valuation_vector = clause_function(x)

            # a = valuation_vector.detach().to("cpu").numpy().reshape(-1, 1)
            clause_function_value_lists.append(valuation_vector)
        # C * B * G
        C = torch.stack(clause_function_value_lists, 0)

        # C = torch.stack([self.cs[i](x) for i in range(self.I.size(0))], 0)

        # taking weighted sum using m weights and stack to a tensor H
        # m * C
        # W_star = torch.softmax(self.W * (1 / self.beta), 1)
        W_star = torch.softmax(self.W, 1)
        # m * C * B * G
        W_tild = W_star.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        # m * C * B * G
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        # m * B * G
        H = torch.sum(W_tild * C_tild, dim=1)
        # taking soft or to compose a logic program with m clauses
        # B * G
        R = softor(H, dim=0, gamma=self.gamma)
        # c = R.detach().to("cpu").numpy().reshape(-1, 1)
        return R

    def r_bk(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # a = x.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        # C * B * G

        value_list = []
        for i in range(self.I_bk.size(0)):
            bk_clause_function = self.cs_bk[i]
            value = bk_clause_function(x)
            # b = value.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
            value_list.append(value)

        C = torch.stack(value_list, 0)
        # B * G
        res = softor(C, dim=0, gamma=self.gamma)
        # b = res.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        return res

    def r_pi(self, x):
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # a = x.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        # C * B * G

        value_list = []
        for i in range(self.I_pi.size(0)):
            pi_clause_function = self.cs_pi[i]
            value = pi_clause_function(x)
            # b = value.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
            value_list.append(value)

        C = torch.stack(value_list, 0)
        # B * G
        res = softor(C, dim=0, gamma=self.gamma)
        # b = res.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        return res


class ClauseInferModule(nn.Module):
    def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk=None, I_pi=None):
        """
        Infer module using each clause.
        The result is not amalgamated in terms of clauses.
        """
        super(ClauseInferModule, self).__init__()
        self.I = I
        self.I_bk = I_bk
        self.I_pi = I_pi
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        if not train:
            self.W = init_identity_weights(I, device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.Tensor(
                np.random.normal(size=(m, I.size(0)))).to(device))
        # clause functions
        self.cs = [ClauseFunction(I[i], gamma=gamma)
                   for i in range(self.I.size(0))]

        if not self.I_bk is None:
            self.cs_bk = [ClauseFunction(I_bk[i], gamma=gamma)
                          for i in range(self.I_bk.size(0))]
        if not I_bk is None:
            self.W_bk = init_identity_weights(I_bk, device)

        if not self.I_pi is None:
            self.cs_pi = [ClauseFunction(I_pi[i], gamma=gamma) for i in range(self.I_pi.size(0))]
            self.W_pi = init_identity_weights(I_pi, device)

        assert m == self.C, "Invalid m and C: " + \
                            str(m) + ' and ' + str(self.C)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        B = x.size(0)
        # C * B * G
        R = x.unsqueeze(dim=0).expand(self.C, B, self.G)
        if self.I_bk is None:
            for t in range(self.infer_step):
                R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        else:
            for t in range(self.infer_step):
                # infer by background knowledge
                # r_bk = self.r_bk(R[0])
                # R_bk = self.r_bk(r_bk).unsqueeze(dim=0).expand(self.C, B, self.G)
                # R = R_bk
                # print("R: ", R.shape)
                # print("r(R): ", self.r(R).shape)
                # print("r_bk(R): ", self.r_bk(R).shape)
                # shape? dim?
                r_R = self.r(R)
                A_A = r_R.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
                r_bk = self.r_bk(R).unsqueeze(dim=0).expand(self.C, B, self.G)
                A_B = r_bk.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
                if self.I_pi is not None:

                    r_pi = self.r_pi(R).unsqueeze(dim=0).expand(self.C, B, self.G)
                    R = softor([R, r_R, r_bk, r_pi], dim=2, gamma=self.gamma)

                    A_C = r_pi.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
                    A_D = R.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
                else:
                    R = softor([R, r_R, r_bk], dim=2, gamma=self.gamma)
        return R

    def r(self, x):
        # x: C * B * G
        B = x.size(1)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # infer from i-th valuation tensor using i-th clause
        C = torch.stack([self.cs[i](x[i]) for i in range(self.I.size(0))], 0)
        return C

    def r_bk(self, x):
        x = x[0]
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # just use the first row
        C = torch.stack([self.cs_bk[i](x)
                         for i in range(self.I_bk.size(0))], 0)
        # B * G
        return softor(C, dim=0, gamma=self.gamma)
        # taking weighted sum using m weights and stack to a tensor H
        # m * C
        W_star = torch.softmax(self.W_bk, 1)
        # m * C * B * G
        W_tild = W_star.unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        # m * C * B * G
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        # m * B * G
        H = torch.sum(W_tild * C_tild, dim=1)
        # taking soft or to compose a logic program with m clauses
        # B * G
        R = softor(H, dim=0, gamma=self.gamma)
        return R

    def r_pi(self, x):
        x = x[0]
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        a = x.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        # C * B * G

        value_list = []
        for i in range(self.I_pi.size(0)):
            pi_clause_function = self.cs_pi[i]
            value = pi_clause_function(x)
            b = value.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
            value_list.append(value)

        C = torch.stack(value_list, 0)
        # B * G
        res = softor(C, dim=0, gamma=self.gamma)
        b = res.detach().to("cpu").numpy().reshape(-1, 1)  # DEBUG
        return res


class PIClauseInferModule(nn.Module):
    def __init__(self, I, infer_step, gamma=0.01, device=None, train=False, m=1, I_bk=None):
        """
        Infer module using each clause.
        The result is not amalgamated in terms of clauses.
        """
        super(PIClauseInferModule, self).__init__()
        self.I = I
        self.I_bk = I_bk
        self.infer_step = infer_step
        self.m = m
        self.C = self.I.size(0)
        self.G = self.I.size(1)
        self.gamma = gamma
        self.device = device
        self.train_ = train
        if not train:
            self.W = init_identity_weights(I, device)
        else:
            # to learng the clause weights, initialize W as follows:
            self.W = nn.Parameter(torch.Tensor(
                np.random.normal(size=(m, I.size(0)))).to(device))
        # clause functions
        self.cs = [ClauseFunction(I[i], gamma=gamma)
                   for i in range(self.I.size(0))]

        if not self.I_bk is None:
            self.cs_bk = [ClauseFunction(I_bk[i], gamma=gamma)
                          for i in range(self.I_bk.size(0))]

        if not I_bk is None:
            self.W_bk = init_identity_weights(I_bk, device)

        assert m == self.C, "Invalid m and C: " + \
                            str(m) + ' and ' + str(self.C)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        B = x.size(0)
        # C * B * G
        R = x.unsqueeze(dim=0).expand(self.C, B, self.G)
        if self.I_bk is None:
            for t in range(self.infer_step):
                R = softor([R, self.r(R)], dim=1, gamma=self.gamma)
        else:
            for t in range(self.infer_step):
                # infer by background knowledge
                # r_bk = self.r_bk(R[0])
                # R_bk = self.r_bk(r_bk).unsqueeze(dim=0).expand(self.C, B, self.G)
                # R = R_bk
                # print("R: ", R.shape)
                # print("r(R): ", self.r(R).shape)
                # print("r_bk(R): ", self.r_bk(R).shape)
                # shape? dim?
                R = softor([R, self.r(R), self.r_bk(R).unsqueeze(
                    dim=0).expand(self.C, B, self.G)], dim=2, gamma=self.gamma)
        return R

    def r(self, x):
        # x: C * B * G
        B = x.size(1)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # infer from i-th valuation tensor using i-th clause
        C = torch.stack([self.cs[i](x[i])
                         for i in range(self.I.size(0))], 0)
        return C

    def r_bk(self, x):
        x = x[0]
        B = x.size(0)  # batch size
        # apply each clause c_i and stack to a tensor C
        # C * B * G
        # just use the first row
        C = torch.stack([self.cs_bk[i](x)
                         for i in range(self.I_bk.size(0))], 0)
        # B * G
        return softor(C, dim=0, gamma=self.gamma)
        # taking weighted sum using m weights and stack to a tensor H
        # m * C
        W_star = torch.softmax(self.W_bk, 1)
        # m * C * B * G
        W_tild = W_star.unsqueeze(
            dim=-1).unsqueeze(dim=-1).expand(self.m, self.C, B, self.G)
        # m * C * B * G
        C_tild = C.unsqueeze(dim=0).expand(self.m, self.C, B, self.G)
        # m * B * G
        H = torch.sum(W_tild * C_tild, dim=1)
        # taking soft or to compose a logic program with m clauses
        # B * G
        R = softor(H, dim=0, gamma=self.gamma)
        return R


class ClauseFunction(nn.Module):
    """
    A class of the clause function.
    """

    def __init__(self, I_i, gamma=0.01):
        super(ClauseFunction, self).__init__()
        # self.i = i  # clause index
        self.I_i = I_i  # index tensor C * S * G, S is the number of possible substituions
        self.L = I_i.size(-1)  # number of body atoms
        self.S = I_i.size(-2)  # max number of possible substitutions
        self.gamma = gamma

    def forward(self, x):
        batch_size = x.size(0)  # batch size
        # B * G
        V = x
        # G * S * b
        # I_i = self.I[self.i, :, :, :]

        # B * G -> B * G * S * L
        V_tild = V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.S, self.L)
        # G * S * L -> B * G * S * L
        I_i_tild = self.I_i.repeat(batch_size, 1, 1, 1)

        # B * G
        C = softor(torch.prod(torch.gather(V_tild, 1, I_i_tild), 3), dim=2, gamma=self.gamma)
        return C
