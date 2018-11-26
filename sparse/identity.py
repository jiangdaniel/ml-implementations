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

import utils
from sparse_layer import SparseLayer


MARGIN = 0.1


def identity(args):
    ns = [8, 16, 32, 64]
    a = [(1, 2), (2, 2), (2, 8), (2, 10)]
    writer = SummaryWriter(os.path.join("./out", args.dir))

    for n, (a_local, a_global) in zip(ns, a):
        if not os.path.exists("./images/{}/".format(args.dir):
            os.makedirs("./images/{}/".format(args.dir))

        cov = th.eye(n)
        if args.correlated:
            for i in range(n-1):
                cov[i, i+1] = 0.5
                cov[i+1, i] = 0.5
        normal = MultivariateNormal(th.zeros(n), cov)
        layer = SparseLayer(n, n, n, a_local, a_global)
        optimizer = optim.Adam(layer.parameters(), args.lr)
        val = normal.sample(th.Size([args.batch_size]))

        running_loss = 0.
        for epoch in range(args.epochs):
            x = th.randn(args.batch_size, n)
            pred = layer(x)
            loss = F.mse_loss(pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if epoch % args.log_iter == args.log_iter - 1:
                with th.no_grad():
                    writer.add_scalar(f"train/{n}dims", running_loss / args.log_iter, epoch)
                    writer.add_scalar(f"val/{n}dims", F.mse_loss(layer(val), val), epoch)
                    running_loss = 0.
                    means = (layer.D.sigmoid() * layer.shape).unsqueeze(0)
                    sigmas = (F.softplus(layer.sigma + layer.sigma_boost) * n * 0.1 + layer.tau).unsqueeze(0).unsqueeze(2).repeat((1, 1, 2))
                    values = layer.v.unsqueeze(0)

                    plt.figure(figsize=(7, 7))
                    plt.cla()
                    utils.plot(means, sigmas, values, shape=(n, n))
                    plt.xlim((-MARGIN*(n-1), (n-1) * (1.0+MARGIN)))
                    plt.ylim((-MARGIN*(n-1), (n-1) * (1.0+MARGIN)))

                    plt.savefig("./images/{}/means{:06}.pdf".format(args.dir, epoch))
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log-iter", type=int, default=100)
    parser.add_argument("--dir", default=utils.timestamp())
    parser.add_argument("--correlated", action="store_true", default=False)

    parser.add_argument("--local", type=int, default=2)
    parser.add_argument("--glob", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    identity(args)
