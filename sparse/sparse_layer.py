#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import normal


class SparseLayer(nn.Module):
    def __init__(self, input_size, output_size, n_gaussians, n_local, n_global, local1=None, local2=None):
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
        self.sigma_boost = 2.
        if local1 is None:
            local1 = local2 = np.log2(input_size)
        self.local_shape = th.Tensor([local1, local2])

        # Parameters
        self.D = nn.Parameter(th.randn(n_gaussians, 2))
        self.sigma = nn.Parameter(th.randn(n_gaussians))
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
        D = self.D.sigmoid() * self.shape
        sigma = F.softplus(self.sigma + self.sigma_boost).unsqueeze(-1).repeat(1, 2) * self.shape * 0.1 + self.tau

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


def main(args):
    x = th.rand(50, 100)
    dim = th.tensor([100., 10.])
    D = th.rand(args.k, 2)
    sigma = th.zeros(args.k).uniform_(-1, 1)
    v = th.zeros(args.k).uniform_(-1, 1)
    l_1 = l_2 = np.log(x.shape[-1])
    local_dim = th.tensor([l_1, l_2]).float()

    D.requires_grad_()
    sigma.requires_grad_()
    v.requires_grad_()

    D_scaled = D.sigmoid() * dim

    with th.no_grad():
        D_prime = th.zeros(args.k * (4 + args.local + args.glob), 2)

        select_nearest = th.tensor([[1., 1.], [0., 1.], [1., 0.], [1., 1.]]).repeat(args.k, 1)
        select_local = (th.rand(args.local * args.k, 2) - 0.5) * local_dim
        D_prime[:4 * args.k] = D_scaled.repeat(1, 4).view(-1, 2) + select_nearest
        D_prime[4 * args.k:(4 + args.local) * args.k] = D_scaled.repeat(1, args.local).view(-1, 2) + select_local
        D_prime[(4 + args.local) * args.k:(4 + args.local + args.glob) * args.k] = th.rand(args.glob * args.k, 2) * dim

        D_prime.round_()
        D_prime[:, 0].clamp_(0, dim[0]-1)
        D_prime[:, 1].clamp_(0, dim[1]-1)

    sigma_scaled = F.softplus(sigma + 2).unsqueeze(-1).repeat(1, 2) * dim * 0.1 + args.tau
    means = D_scaled.t().unsqueeze(0).repeat(args.k * (4 + args.local + args.glob), 1, 1)
    stds = sigma_scaled.sqrt().t().unsqueeze(0).repeat(args.k * (4 + args.local + args.glob), 1, 1)
    z = (D_prime.unsqueeze(-1).repeat(1, 1, 3) - means) / stds

    m = normal.Normal(th.tensor([0.0]), th.tensor([1.0]))
    probs = m.log_prob(z).sum(1).exp()

    # Remove duplicates using Cantor technique described in paper
    with th.no_grad():
        cantor = D_prime.sum(1) * (D_prime.sum(1) + 1) / 2 + D_prime[:, 1]
        cantor_sort, cantor_indices = cantor.sort()
        mask = th.cat((th.zeros(1, dtype=th.uint8), cantor_sort[1:] == cantor_sort[:-1]))[cantor_indices]
    probs = probs * (1-mask).unsqueeze(-1).float()
    probs_intermediate = probs / probs.sum(0, keepdim=True)
    v_prime = (probs_intermediate * v).sum(1)
    out = mm(x, D_prime.long(), v_prime, th.Size([100, 10]))
    loss = out.sum()
    loss.backward()


def mm(input, indices, values, shape):
    output = th.zeros(input.shape[0], shape[1])
    for (r, c), val in zip(indices, values):
        output[:, c] += val * input[:, r]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--local", type=int, default=2)
    parser.add_argument("--glob", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
