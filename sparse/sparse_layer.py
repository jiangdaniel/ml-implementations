#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import normal


EPSILON = 10e-7
SIGMA_BOOST = 2.0


class SparseLayer(nn.Module):
    def __init__(self, input_size, output_size, n_gaussians, n_local, n_global, fix_values=False, local1=None, local2=None):
        super(SparseLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.shape = th.Tensor([input_size, output_size])
        self.standard = normal.Normal(th.tensor([0.]), th.tensor([1.]))

        # Hyperparameters
        self.n_gauss = n_gaussians
        self.n_local = n_local
        self.n_global = n_global
        self.tau = 0.1
        if local1 is None:
            local1 = local2 = 4.
        self.local_shape = th.Tensor([local1, local2])

        # Parameters
        self.D = nn.Parameter(th.randn(n_gaussians, 2))
        self.sigma = nn.Parameter(th.randn(n_gaussians))
        if fix_values:
            self.v = th.ones(n_gaussians)
        else:
            self.v = nn.Parameter(th.randn(n_gaussians))

    def forward(self, x):
        indices, values = self._sample_weight()
        out = self._sparse_mm(x, indices, values)
        return out

    def _sparse_mm(self, x, indices, values):
        """Multiply sparse weights and dense input to get a dense output"""
        output = th.zeros(x.shape[0], self.output_size)
        for (r, c), val in zip(indices, values):
            output[:, c] += val * x[:, r]
        return output

    def _sample_weight(self):
        D = self.D.sigmoid() * (self.shape - 1)
        sigma = (F.softplus(self.sigma + SIGMA_BOOST) + EPSILON).unsqueeze(-1).repeat(1, 2) * self.shape * 0.1 + self.tau

        with th.no_grad():
            D_prime = th.zeros(self.n_gauss * (4 + self.n_local + self.n_global), 2)

            select_nearest = th.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).repeat(self.n_gauss, 1)
            select_local = (th.rand(self.n_local * self.n_gauss, 2) - 0.5) * self.local_shape
            D_prime[:4 * self.n_gauss] = D.repeat(1, 4).view(-1, 2) + select_nearest
            D_prime[4 * self.n_gauss:(4 + self.n_local) * self.n_gauss] = D.repeat(1, self.n_local).view(-1, 2) + select_local
            D_prime[(4 + self.n_local) * self.n_gauss:(4 + self.n_local + self.n_global) * self.n_gauss] = th.rand(self.n_global * self.n_gauss, 2) * self.shape

            D_prime.round_()
            D_prime[:, 0].clamp_(0, self.shape[0]-1)
            D_prime[:, 1].clamp_(0, self.shape[1]-1)

        means = D.t().unsqueeze(0).repeat(self.n_gauss * (4 + self.n_local + self.n_global), 1, 1)
        stds = sigma.sqrt().t().unsqueeze(0).repeat(self.n_gauss * (4 + self.n_local + self.n_global), 1, 1)
        z = (D_prime.unsqueeze(-1).repeat(1, 1, self.n_gauss) - means) / stds

        probs = (self.standard.log_prob(z) - stds.log()).sum(1).exp()

        # Remove duplicates using Cantor technique described in paper
        with th.no_grad():
            cantor = D_prime.sum(1) * (D_prime.sum(1) + 1) / 2 + D_prime[:, 1]
            cantor_sort, cantor_indices = cantor.sort()
            mask = th.cat((th.zeros(1, dtype=th.uint8), cantor_sort[1:] == cantor_sort[:-1]))[cantor_indices]
        probs = probs * (1-mask).unsqueeze(-1).float()
        probs_intermediate = probs / probs.sum(0, keepdim=True)
        v_prime = (probs_intermediate * self.v).sum(1)
        return D_prime.long(), v_prime
