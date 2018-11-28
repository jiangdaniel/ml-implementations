#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils
from sparse_layer import (
    SparseLayer,
    SIGMA_BOOST,
    EPSILON)


MARGIN = 0.1


def identity(args):
    ns = [8, 16, 32, 64]
    a = [(1, 2), (2, 2), (2, 8), (2, 10)]
    writer = SummaryWriter(os.path.join("./logs", args.dir))

    for n, (a_local, a_global) in zip(ns, a):
        if not os.path.exists(f"./images/{args.dir}/{n}"):
            os.makedirs(f"./images/{args.dir}/{n}")

        cov = th.eye(n)
        if args.correlated:
            for i in range(n-1):
                cov[i, i+1] = 0.5
                cov[i+1, i] = 0.5
        normal = MultivariateNormal(th.zeros(n), cov)
        layer = SparseLayer(n, n, n, a_local, a_global, fix_values=True).to(device)
        optimizer = optim.Adam(layer.parameters(), args.lr)
        val = normal.sample(th.Size([args.batch_size])).to(device)

        running_loss = 0.
        for epoch in tqdm(range(args.epochs)):
            x = th.randn(args.batch_size, n, device=device)
            pred = layer(x)
            loss = F.mse_loss(pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if epoch % args.log_iter == args.log_iter - 1:
                with th.no_grad():
                    writer.add_scalar(f"train/{n}dims", running_loss / args.log_iter, epoch)
                    writer.add_scalar(f"val/{n}dims", F.mse_loss(layer(val), val).item(), epoch)
                    running_loss = 0.
                    means, sigmas, values = layer.hyper()
                    means, sigmas, values = means.unsqueeze(0), sigmas.unsqueeze(0), values.unsqueeze(0)

                    plt.figure(figsize=(7, 7))
                    plt.cla()
                    utils.plot(means, sigmas, values, shape=(n, n))
                    plt.xlim((-MARGIN*(n-1), (n-1) * (1.0+MARGIN)))
                    plt.ylim((-MARGIN*(n-1), (n-1) * (1.0+MARGIN)))

                    plt.savefig("./images/{}/{}/means{:06}.pdf".format(args.dir, n, epoch))
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log-iter", type=int, default=1000)
    parser.add_argument("--dir", default=utils.timestamp())
    parser.add_argument("--correlated", action="store_true", default=False)

    parser.add_argument("--local", type=int, default=4)
    parser.add_argument("--glob", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()
    device = th.device("cpu" if (not th.cuda.is_available() or args.no_cuda) else "cuda")
    device = th.device("cpu")  # disables gpu support
    identity(args)
