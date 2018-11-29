#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import torch as th
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils


def main(args):
    writer = SummaryWriter(os.path.join("logs", args.dir))
    trainloader, testloader = get_loaders(args.batch_size)
    epsilon = 0.5
    beta = 1.0
    alpha1 = 0.1
    alpha2 = 0.05

    a = np.sqrt(2.0 / (784 + 500))
    W1 = th.randn(784, 500, device=device) * a
    b1 = th.randn(500, device=device) * a

    a = np.sqrt(2.0 / (500 + 10))
    W2 = th.randn(500, 10, device=device) * a
    b2 = th.randn(10, device=device) * a

    states = [(th.rand(args.batch_size, 500, device=device), \
            th.rand(args.batch_size, 10, device=device)) for _ in range(len(trainloader))]

    for epoch in range(args.epochs):
        running_loss = running_energy = running_true_positive = 0.

        for i, (x, labels) in enumerate(tqdm(trainloader)):
            x, labels = x.view(-1, 784).to(device), labels.to(device)
            h, y = states[i]

            # Free phase
            for j in range(20):
                dh = d_rho(h) * (x @ W1 + y @ W2.t() + b1) - h
                dy = d_rho(y) * (h @ W2 + b2) - y

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)
                '''
                energy = (h.pow(2).sum() + y.pow(2).sum() \
                    - (W1 * (x.t() @ h)).sum() - (W2 * (h.t() @ y)).sum() \
                    - 2 * (h @ b1).sum() - 2 * (y @ b2).sum()).item() / 2
                print(energy, dh.norm().item())
                '''

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
                '''
                energy = (h.pow(2).sum() + y.pow(2).sum() \
                    - (W1 * (x.t() @ h)).sum() - (W2 * (h.t() @ y)).sum() \
                    - 2 * (h @ b1).sum() - 2 * (y @ b2).sum()).item() / 2
                print(energy, dh.norm().item())
                '''

            h_clamped = h
            y_clamped = y
            W1 += alpha1 / beta * (rho(x.t()) @ (rho(h_clamped) - rho(h_free))) / args.batch_size
            W2 += alpha2 / beta * (rho(h_clamped.t()) @ rho(y_clamped) - rho(h_free.t()) @ rho(y_free)) / args.batch_size
            b1 += alpha1 / beta * (rho(h_clamped) - rho(h_free)).mean(0)
            b2 += alpha2 / beta * (rho(y_clamped) - rho(y_free)).mean(0)

            running_energy += (h_free.pow(2).sum() / 2 + y_free.pow(2).sum() \
                - ((W1 * (x.t() @ h_free)) / 2).sum() - ((W2 * (h_free.t() @ y_free)) / 2).sum() \
                - (h_free @ b1).sum() - (y_free @ b2).sum()).item()
            running_loss += (t - y_free).pow(2).sum().item()
            running_true_positive += (y_free.argmax(1) == labels).sum().item()

        energy_train = running_energy / (len(trainloader) * args.batch_size)
        accuracy_train = running_true_positive / (len(trainloader) * args.batch_size)
        loss_train = running_loss / (len(trainloader) * args.batch_size)
        print(f"Energy: {energy_train}, Accuracy: {accuracy_train}, Loss: {loss_train}")
        writer.add_scalar(f"loss", loss_train, epoch)
        writer.add_scalar(f"energy", energy_train, epoch)
        writer.add_scalar(f"accuracy", accuracy_train, epoch)

def rho(x):
    return x.clamp(0., 1.)

def d_rho(x):
    return ((x >= 0.) * (x <= 1.)).float()

def get_loaders(batch_size, fashion=False):
    mnist = torchvision.datasets.MNIST
    if fashion:
        mnist = torchvision.datasets.FashionMNIST

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),])

    trainloader = th.utils.data.DataLoader(
        mnist(root="./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    testloader = th.utils.data.DataLoader(
        mnist(root="./data", train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return trainloader, testloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=20)

    parser.add_argument("--fashion", action="store_true", default=False,
        help="use fashion mnist")
    parser.add_argument("--dir", default=utils.timestamp(),
        help="name of output log directory")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()
    device = th.device("cpu" if (not th.cuda.is_available() or args.no_cuda) else "cuda")
    main(args)
