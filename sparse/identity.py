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
from tensorboardX import SummaryWriter

from sparse import SparseLayer


def identity(args):
    ns = [8, 16, 32, 64]
    a = [(1, 2), (2, 2), (2, 8), (2, 10)]
    writer = SummaryWriter("./out")

    for n, (a_local, a_global) in zip(ns, a):
        layer = SparseLayer(n, n, args.k, a_local, a_global)
        optimizer = optim.Adam(layer.parameters(), args.lr)
        val = th.randn(args.batch_size, n)

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
                writer.add_scalar(f"train/{n}", running_loss / args.log_iter, epoch)
                writer.add_scalar(f"val/{n}", F.mse_loss(layer(val), val), epoch)
                running_loss = 0.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log-iter", type=int, default=20)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--local", type=int, default=2)
    parser.add_argument("--glob", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    args = parser.parse_args()
    identity(args)
