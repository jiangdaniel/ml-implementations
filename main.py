#!/usr/bin/env python3

import argparse

import numpy as np
import torch as th


def main(args):
    x = th.zeros(50, 100)
    dim = th.tensor([100, 10]).float()
    D = th.rand(args.k, 2)
    sigma = th.zeros(args.k).uniform_(-1, 1)
    v = th.zeros(args.k).uniform_(-1, 1)
    l_1 = l_2 = np.log(x.shape[-1])
    local_dim = th.tensor([l_1, l_2]).float()

    D_scaled = D.sigmoid() * dim

    with th.no_grad():
        D_hat = th.zeros(args.k * (4 + args.local + args.glob), 2)

        select_nearest = th.tensor([[1, 1], [0, 1], [1, 0], [1, 1]]).repeat(args.k, 1).float()
        select_local = (th.rand(args.local * args.k, 2) - 0.5) * local_dim
        D_hat[:4 * args.k] = D_scaled.repeat(1, 4).view(-1, 2) + select_nearest
        D_hat[4 * args.k:(4 + args.local) * args.k] = D_scaled.repeat(1, args.local).view(-1, 2) + select_local
        D_hat[(4 + args.local) * args.k:(4 + args.local + args.glob) * args.k] = th.rand(args.glob * args.k, 2) * dim

        D_hat.round_()
        D_hat[:, 0].clamp_(0, dim[0]-1)
        D_hat[:, 1].clamp_(0, dim[1]-1)

    v_hat = th.zeros(args.k * (4 + args.local + args.glob))
    sigma_i = (sigma[i] + 2).softplus() * dim * 0.1 + args.tau


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--local", type=int, default=2)
    parser.add_argument("--glob", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
