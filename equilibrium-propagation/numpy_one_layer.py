#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch as th
import torchvision
from tqdm import tqdm


def main(args):
    trainloader, testloader = get_loaders(args.batch_size, args.fashion)
    epsilon = 0.5
    beta = 1.0
    alpha1 = 0.1
    alpha2 = 0.05

    a = np.sqrt(2.0 / (784 + 500))
    W1 = np.random.uniform(-a, a, (784, 500))
    b1 = np.random.uniform(-a, a, 500)

    a = np.sqrt(2.0 / (500 + 10))
    W2 = np.random.uniform(-a, a, (500, 10))
    b2 = np.random.uniform(-a, a, 10)

    states = [(np.random.uniform(0, 1., (args.batch_size, 500)), \
            np.random.uniform(0, 1., (args.batch_size, 10))) for _ in range(len(trainloader))]

    for epoch in range(args.epochs):
        running_loss = running_energy = running_true_positive = 0.
        for i, (x, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch}")):
            x, labels = x.view(-1, 784).numpy(), labels.numpy()
            h, y = states[i]

            # Free phase
            for j in range(20):
                dh = d_rho(h) * (x @ W1 + y @ W2.T + b1) - h
                dy = d_rho(y) * (h @ W2 + b2) - y

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)
                '''
                energy = (np.square(h).sum() + np.square(y).sum() \
                    - (W1 * (x.T @ h)).sum() - (W2 * (h.T @ y)).sum()) / 2 \
                    - (h @ b1).sum() - (y @ b2).sum())
                print(np.round(energy, 4), np.round(np.linalg.norm(dh), 4))
                '''

            h_free, y_free = np.copy(h), np.copy(y)
            states[i] = h_free, y_free

            t = np.zeros((x.shape[0], 10))
            t[np.arange(t.shape[0]), labels] = 1

            # Weakly clamped phase
            for j in range(4):
                dh = d_rho(h) * (x @ W1 + y @ W2.T + b1) - h
                dy = d_rho(y) * (h @ W2 + b2) - y + beta * (t - y)

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)
                '''
                energy = (np.square(h).sum() + np.square(y).sum() \
                    - (W1 * (x.T @ h)).sum() - (W2 * (h.T @ y)).sum()) / 2 \
                    - (h @ b1).sum() - (y @ b2).sum()
                print(np.round(energy, 4), np.round(np.linalg.norm(dh), 4))
                '''

            h_clamped = np.copy(h)
            y_clamped = np.copy(y)

            W1 += alpha1 / beta * (rho(x.T) @ rho(h_clamped) - rho(x.T) @ rho(h_free)) / args.batch_size
            W2 += alpha2 / beta * (rho(h_clamped.T) @ rho(y_clamped) - rho(h_free.T) @ rho(y_free)) / args.batch_size
            b1 += alpha1 / beta * (rho(h_clamped) - rho(h_free)).mean(0)
            b2 += alpha2 / beta * (rho(y_clamped) - rho(y_free)).mean(0)

            running_energy += (np.square(h_free).sum() + np.square(y_free).sum() \
                - (W1 * (x.T @ h_free)).sum() - (W2 * (h_free.T @ y_free)).sum()) / 2 \
                - (h_free @ b1).sum() - (y_free @ b2).sum()
            running_loss += np.square(t - y_free).sum()
            running_true_positive += np.count_nonzero(np.argmax(y_free, 1) == labels)

        energy_avg = running_energy / (len(trainloader) * args.batch_size)
        accuracy_avg = running_true_positive / (len(trainloader) * args.batch_size)
        loss_avg = running_loss / (len(trainloader) * args.batch_size)
        print(f"Energy: {energy_avg}, Accuracy: {accuracy_avg}, Loss: {loss_avg}")

def rho(x):
    return np.copy(np.clip(x, 0., 1.))

def d_rho(x):
    return (x >= 0.) * (x <= 1.)

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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)

    parser.add_argument("--fashion", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
