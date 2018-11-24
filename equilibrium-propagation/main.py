#!/usr/bin/env python3

import os
import argparse

import numpy as np
import torch as th
import torchvision
from tqdm import tqdm
import tensorboardX


def main(args):
    trainloader = get_trainloader(args.batch_size)
    epsilon = 0.5
    beta = 1.0
    alpha1 = 0.1
    alpha2 = 0.05

    a = np.sqrt(2.0 / (784 + 500))
    W1 = th.randn(784, 500, device=device) * a
    b1 = th.randn(500, device=device) * a/2

    a = np.sqrt(2.0 / (500 + 10))
    W2 = th.randn(500, 10, device=device) * a
    b2 = th.randn(10, device=device) * a/2

    running_loss = 0.
    running_energy = 0.
    running_true_positive = 0.
    states = []
    for _ in range(len(trainloader)):
        h = th.rand(args.batch_size, 500)
        y = th.rand(args.batch_size, 10)
        states.append((h, y))

    for epoch in range(args.epochs):
        for i, (x, labels) in enumerate(tqdm(trainloader)):
            x, labels = x.view(-1, 784).to(device), labels.to(device)
            h, y = states[i]

            # Free phase
            for j in range(20):
                dh = d_rho(h) * (x @ W1 + y @ W2.t() + b1) - h
                dy = d_rho(y) * (h @ W2 + b2) - y

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)

            h_free, y_free = h, y
            states[i] = h_free, y_free

            t = th.zeros(x.shape[0], 10, device=device)
            t.scatter_(1, labels.unsqueeze(1), 1)

            # Weakly clamped
            for j in range(4):
                dy = d_rho(y) * (h @ W2 + b2) - y + beta * (t - y)
                dh = d_rho(h) * (x @ W1 + y @ W2.t() + b1) - h

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)

            h_clamped = h
            y_clamped = y

            W1 += alpha1 * (1/beta) * (rho(x.t()) @ rho(h_clamped) - rho(x.t()) @ rho(h_free))
            W2 += alpha2 * (1/beta) * (rho(h_clamped.t()) @ rho(y_clamped) - rho(h_free.t()) @ rho(y_free))

            running_energy += (h_free.pow(2).sum() / 2 + y_free.pow(2).sum() \
                - ((W1 * (x.t() @ h)) / 2).sum() - ((W2 * (h.t() @ y)) / 2).sum() \
                - (h @ b1).sum() - (y @ b2).sum()).item()
            running_loss += (t - y_free).pow(2).sum().item()
            running_true_positive += (y_free.max(1)[0].long() == labels).sum().item()
            if i % args.log_iter == args.log_iter - 1:
                print(f"Energy: {running_energy / args.batch_size / args.log_iter}, Accuracy: {running_true_positive / args.batch_size / args.log_iter}, Loss: {running_loss / args.log_iter}")
                running_loss = 0.
                running_energy = 0.
                running_true_positive = 0.

def rho(x):
    return x.clamp(0., 1.)

def d_rho(x):
    # greater than or equal to?
    return ((x >= 0.) * (x <= 1.)).float()

def get_trainloader(batch_size):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),])
    traindataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform)
    trainloader = th.utils.data.DataLoader(
        traindataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    return trainloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log-iter", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()
    device = th.device("cpu" if (not th.cuda.is_available() or args.no_cuda) else "cuda")
    main(args)
