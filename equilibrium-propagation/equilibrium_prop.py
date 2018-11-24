#!/usr/bin/env python3

import argparse

import numpy as np
import torch as th
import torchvision
from tqdm import tqdm
import ipdb


def main(args):
    trainloader = get_trainloader(args.batch_size)
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

    running_loss = 0.
    running_energy = 0.
    running_true_positive = 0.
    states = []
    for _ in range(len(trainloader)):
        h = np.random.uniform(0, 1., (args.batch_size, 500))
        y = np.random.uniform(0, 1., (args.batch_size, 10))
        states.append((h, y))

    for epoch in range(args.epochs):
        for i, (x, labels) in enumerate(tqdm(trainloader)):
            x, labels = x.view(-1, 784).numpy(), labels.numpy()  # TODO: standardize?
            #x = x * 2. - 1.
            h, y = states[i]

            # Free phase
            for j in range(20):
                dh = d_rho(h) * (x @ W1 + y @ W2.T + b1) - h
                dy = d_rho(y) * (h @ W2 + b2) - y

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)
                energy = np.square(h).sum() / 2 + np.square(y).sum() / 2 \
                    - ((W1 * (x.T @ h)) / 2).sum() - ((W2 * (h.T @ y)) / 2).sum() \
                    - (h @ b1).sum() - (y @ b2).sum()
                #print(np.round(energy, 4), np.round(np.linalg.norm(dh), 4))
            #import ipdb; ipdb.set_trace()

            h_free, y_free = np.copy(h), np.copy(y)
            states[i] = h_free, y_free

            t = np.zeros((x.shape[0], 10))
            t[np.arange(t.shape[0]), labels] = 1

            # Weakly clamped
            for j in range(4):
                dh = d_rho(h) * (x @ W1 + y @ W2.T + b1) - h
                dy = d_rho(y) * (h @ W2 + b2) - y + beta * (t - y)

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)
                energy = np.square(h).sum() / 2 + np.square(y).sum() / 2 \
                    - (W1 * (x.T @ h)).sum() / 2 - (W2 * (h.T @ y)).sum() / 2 \
                    - (h @ b1).sum() - (y @ b2).sum() + beta * np.square(y - t).sum() / 2
                #print(np.round(energy, 4), np.round(np.linalg.norm(dh), 4))
            #import ipdb; ipdb.set_trace()

            h_clamped = np.copy(h)
            y_clamped = np.copy(y)

            W1 += alpha1 / beta * (rho(x.T) @ rho(h_clamped) - rho(x.T) @ rho(h_free)) / args.batch_size
            W2 += alpha2 / beta * (rho(h_clamped.T) @ rho(y_clamped) - rho(h_free.T) @ rho(y_free)) / args.batch_size
            b1 += alpha1 / beta * (rho(h_clamped) - rho(h_free)).mean(0)
            b2 += alpha2 / beta * (rho(y_clamped) - rho(y_free)).mean(0)

            running_energy += np.square(h_free).sum() / 2 + np.square(y_free).sum() / 2 \
                - ((W1 * (x.T @ h_free)) / 2).sum() - ((W2 * (h_free.T @ y_free)) / 2).sum() \
                - (h_free @ b1).sum() - (y_free @ b2).sum()
            running_loss += np.square(t - y_free).sum()
            running_true_positive += np.count_nonzero(np.argmax(y_free, 1) == labels)
            if i % args.log_iter == args.log_iter - 1:
                print(f"Energy: {running_energy / args.batch_size / args.log_iter}, Accuracy: {running_true_positive / args.batch_size / args.log_iter}, Loss: {running_loss / args.log_iter}")
                running_loss = 0.
                running_energy = 0.
                running_true_positive = 0.

def rho(x):
    return np.copy(np.clip(x, 0., 1.))

def d_rho(x):
    # greater than or equal to?
    return (x >= 0.) * (x <= 1.)


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
    args = parser.parse_args()
    main(args)
