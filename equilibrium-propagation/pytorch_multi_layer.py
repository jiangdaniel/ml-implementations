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
from networks.network import Net

def main(args):
    writer = SummaryWriter(os.path.join("logs", args.dir))
    trainloader, testloader = get_loaders(args.batch_size)

    net = Net(n_layers=1)
    states = []
    for x, labels in trainloader:
        states.append([th.rand(args.batch_size, 500, device=device), th.rand(args.batch_size, 10, device=device)])

    for epoch in range(args.epochs):
        running_loss = running_energy = running_true_positive = 0.

        for i, (x, labels) in enumerate(tqdm(trainloader)):
            x, labels = x.to(device).view(x.shape[0], -1), labels.to(device)
            t = th.zeros(x.shape[0], 10, device=device)
            t.scatter_(1, labels.unsqueeze(1), 1)

            units = [x] + states[i]

            units_free, units_clamped = net.fixed_points(units, t)
            states[i] = units_free[1:]

            net.update(units_free, units_clamped)

            running_true_positive += (units_free[-1].argmax(1) == labels).sum().item()
            running_loss += (t - units_free[-1]).pow(2).sum().item()

        energy_train = running_energy / (len(trainloader) * args.batch_size)
        accuracy_train = running_true_positive / (len(trainloader) * args.batch_size)
        loss_train = running_loss / (len(trainloader) * args.batch_size)
        print(f"Energy: {energy_train}, Accuracy: {accuracy_train}, Loss: {loss_train}")
        writer.add_scalar(f"loss", loss_train, epoch)
        writer.add_scalar(f"energy", energy_train, epoch)
        writer.add_scalar(f"accuracy", accuracy_train, epoch)

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
    device = th.device("cpu")  # gpu version not working
    main(args)
