#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import torch as th
import torch.optim as optim
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter

import network
import utils


def main(args):
    trainloader, testloader = get_loaders(args.batch_size, args.fashion)
    writer = SummaryWriter(os.path.join("logs", args.dir))
    net = network.BayesianNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = running_true_positive = 0.
        for i, (x, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch}")):
            x, labels = x.view(-1, 784).to(device), labels.to(device)
            pred, weights, biases = net.forward(x)
            prior = net.log_likelihood_prior(weights, biases)
            posterior = net.log_likelihood_posterior(weights, biases)

            t = th.zeros(x.shape[0], 10, device=device)
            t.scatter_(1, labels.unsqueeze(1), 1)

            log_likelihood_data = (t * pred).sum()

            loss = (posterior - prior) / len(trainloader) - log_likelihood_data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_true_positive += (pred.argmax(1) == labels).sum().item()
            running_loss += loss.item()

            if i % args.log_iter == args.log_iter - 1:
                accuracy_train = running_true_positive / (args.batch_size * args.log_iter)
                loss_train = running_loss / (args.batch_size * args.log_iter)
                print(f"Train loss: {loss_train}, Train accuracy: {accuracy_train}")
                running_true_positive = running_loss = 0.


def get_loaders(batch_size, fashion=False):
    mnist = torchvision.datasets.MNIST
    if fashion:
        mnist = torchvision.datasets.MNIST

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=int, default=0.01)
    parser.add_argument("--log-iter", type=int, default=10)

    parser.add_argument("--fashion", action="store_true", default=False)
    parser.add_argument("--dir", default=utils.timestamp())
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()
    device = th.device("cpu" if (not th.cuda.is_available() or args.no_cuda) else "cuda")
    main(args)
