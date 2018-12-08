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
    states_test = [(th.rand(args.batch_size, 500, device=device), \
            th.rand(args.batch_size, 10, device=device)) for _ in range(len(testloader))]

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

        running_true_positive = 0.
        for i, (x, labels) in enumerate(tqdm(testloader)):
            x, labels = x.view(-1, 784).to(device), labels.to(device)
            h, y = states_test[i]

            # Free phase
            for j in range(20):
                dh = d_rho(h) * (x @ W1 + y @ W2.t() + b1) - h
                dy = d_rho(y) * (h @ W2 + b2) - y

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)

            h_free, y_free = h, y
            states_test[i] = h_free, y_free

            running_true_positive += (y_free.argmax(1) == labels).sum().item()
        accuracy_test = running_true_positive / (len(testloader) * args.batch_size)
        writer.add_scalar(f"accuracy_test", accuracy_test, epoch)
        print(f"Energy: {energy_train}, Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}, Loss: {loss_train}")

    print("NEXT LAYER")

    a = np.sqrt(2.0 / (500 + 500))
    W2 = th.randn(500, 500, device=device) * a
    b2 = th.randn(500, device=device) * a

    a = np.sqrt(2.0 / (500 + 10))
    W3 = th.randn(500, 10, device=device) * a
    b3 = th.randn(10, device=device) * a

    for i in range(len(states)):
        states[i] = (states[i][0], th.rand(args.batch_size, 500, device=device), \
            th.rand(args.batch_size, 10, device=device))
    for i in range(len(states_test)):
        states_test[i] = (states_test[i][0], \
            th.rand(args.batch_size, 500, device=device),\
            th.rand(args.batch_size, 10, device=device))

    for epoch in range(args.epochs):
        running_loss = running_energy = running_true_positive = 0.

        for i, (_, labels) in enumerate(tqdm(trainloader)):
            labels = labels.to(device)
            x, h, y = states[i]

            # Free phase
            for j in range(20):
                dh = d_rho(h) * (x @ W2 + y @ W3.t() + b2) - h
                dy = d_rho(y) * (h @ W3 + b3) - y

                h = rho(h + epsilon * dh)
                y = rho(y + epsilon * dy)
                '''
                energy = (h.pow(2).sum() + y.pow(2).sum() \
                    - (W1 * (x.t() @ h)).sum() - (W2 * (h.t() @ y)).sum() \
                    - 2 * (h @ b1).sum() - 2 * (y @ b2).sum()).item() / 2
                print(energy, dh.norm().item())
                '''

            h_free, y_free = h, y
            states[i] = x, h_free, y_free

            t = th.zeros(x.shape[0], 10, device=device)
            t.scatter_(1, labels.unsqueeze(1), 1)

            # Weakly clamped
            for j in range(4):
                dy = d_rho(y) * (h @ W3 + b3) - y + beta * (t - y)
                dh = d_rho(h) * (x @ W2 + y @ W3.t() + b2) - h

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
            W2 += alpha1 / beta * (rho(x.t()) @ (rho(h_clamped) - rho(h_free))) / args.batch_size
            W3 += alpha2 / beta * (rho(h_clamped.t()) @ rho(y_clamped) - rho(h_free.t()) @ rho(y_free)) / args.batch_size
            b2 += alpha1 / beta * (rho(h_clamped) - rho(h_free)).mean(0)
            b3 += alpha2 / beta * (rho(y_clamped) - rho(y_free)).mean(0)

            running_energy += (h_free.pow(2).sum() / 2 + y_free.pow(2).sum() \
                - ((W2 * (x.t() @ h_free)) / 2).sum() - ((W3 * (h_free.t() @ y_free)) / 2).sum() \
                - (h_free @ b2).sum() - (y_free @ b3).sum()).item()
            running_loss += (t - y_free).pow(2).sum().item()
            running_true_positive += (y_free.argmax(1) == labels).sum().item()

        energy_train = running_energy / (len(trainloader) * args.batch_size)
        accuracy_train = running_true_positive / (len(trainloader) * args.batch_size)
        loss_train = running_loss / (len(trainloader) * args.batch_size)
        print(f"Energy: {energy_train}, Accuracy: {accuracy_train}, Loss: {loss_train}")
        writer.add_scalar(f"loss", loss_train, epoch + args.epochs)
        writer.add_scalar(f"energy", energy_train, epoch + args.epochs)
        writer.add_scalar(f"accuracy", accuracy_train, epoch + args.epochs)

        running_true_positive = 0.
        for i, (x, labels) in enumerate(tqdm(testloader)):
            x, labels = x.view(-1, 784).to(device), labels.to(device)
            h1, h2, y = states_test[i]

            # Free phase
            for j in range(20):
                #dh1 = d_rho(h1) * (x @ W1 + h2 @ W2.t() + b1) - h1
                dh2 = d_rho(h2) * (h1 @ W2 + y @ W3.t() + b2) - h2
                dy = d_rho(y) * (h2 @ W3 + b3) - y

                #h1 = rho(h1 + epsilon * dh1)
                h2 = rho(h2 + epsilon * dh2)
                y = rho(y + epsilon * dy)

            h1_free, h2_free, y_free = h1, h2, y
            states_test[i] = h1_free, h2_free, y_free

            running_true_positive += (y_free.argmax(1) == labels).sum().item()
        accuracy_test = running_true_positive / (len(testloader) * args.batch_size)
        writer.add_scalar(f"accuracy_test", accuracy_test, epoch + args.epochs)
        print(f"Energy: {energy_train}, Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}, Loss: {loss_train}")

    print("FINAL STAGE")
    alpha1 = 0.4
    alpha2 = 0.1
    alpha3 = 0.01
    for epoch in range(args.epochs_final):
        running_loss = running_energy = running_true_positive = 0.

        for i, (x, labels) in enumerate(tqdm(trainloader)):
            x, labels = x.view(-1, 784).to(device), labels.to(device)
            h1, h2, y = states[i]

            # Free phase
            for j in range(100):
                dh1 = d_rho(h1) * (x @ W1 + h2 @ W2.t() + b1) - h1
                dh2 = d_rho(h2) * (h1 @ W2 + y @ W3.t() + b2) - h2
                dy = d_rho(y) * (h2 @ W3 + b3) - y

                h1 = rho(h1 + epsilon * dh1)
                h2 = rho(h2 + epsilon * dh2)
                y = rho(y + epsilon * dy)
                '''
                energy = (h.pow(2).sum() + y.pow(2).sum() \
                    - (W1 * (x.t() @ h)).sum() - (W2 * (h.t() @ y)).sum() \
                    - 2 * (h @ b1).sum() - 2 * (y @ b2).sum()).item() / 2
                print(energy, dh.norm().item())
                '''

            h1_free, h2_free, y_free = h1, h2, y
            states[i] = h1_free, h2_free, y_free

            t = th.zeros(x.shape[0], 10, device=device)
            t.scatter_(1, labels.unsqueeze(1), 1)

            # Weakly clamped
            for j in range(6):
                dy = d_rho(y) * (h2 @ W3 + b3) - y + beta * (t - y)
                dh2 = d_rho(h2) * (h1 @ W2 + y @ W3.t() + b2) - h2
                dh1 = d_rho(h1) * (x @ W1 + h2 @ W2.t() + b1) - h1

                h1 = rho(h1 + epsilon * dh1)
                h2 = rho(h2 + epsilon * dh2)
                y = rho(y + epsilon * dy)
                '''
                energy = (h.pow(2).sum() + y.pow(2).sum() \
                    - (W1 * (x.t() @ h)).sum() - (W2 * (h.t() @ y)).sum() \
                    - 2 * (h @ b1).sum() - 2 * (y @ b2).sum()).item() / 2
                print(energy, dh.norm().item())
                '''

            h1_clamped = h1
            h2_clamped = h2
            y_clamped = y
            W1 += alpha1 / beta * (rho(x.t()) @ (rho(h1_clamped) - rho(h1_free))) / args.batch_size
            W2 += alpha2 / beta * (rho(h1_clamped.t()) @ rho(h2_clamped) - rho(h1_free.t()) @ rho(h2_free)) / args.batch_size
            W3 += alpha3 / beta * (rho(h2_clamped.t()) @ rho(y_clamped) - rho(h2_free.t()) @ rho(y_free)) / args.batch_size
            b1 += alpha1 / beta * (rho(h1_clamped) - rho(h1_free)).mean(0)
            b2 += alpha2 / beta * (rho(h2_clamped) - rho(h2_free)).mean(0)
            b3 += alpha3 / beta * (rho(y_clamped) - rho(y_free)).mean(0)

            running_energy += (h2_free.pow(2).sum() / 2 + y_free.pow(2).sum() \
                - ((W2 * (h1_free.t() @ h2_free)) / 2).sum() - ((W3 * (h2_free.t() @ y_free)) / 2).sum() \
                - (h2_free @ b2).sum() - (y_free @ b3).sum()).item()
            running_loss += (t - y_free).pow(2).sum().item()
            running_true_positive += (y_free.argmax(1) == labels).sum().item()

        energy_train = running_energy / (len(trainloader) * args.batch_size)
        accuracy_train = running_true_positive / (len(trainloader) * args.batch_size)
        loss_train = running_loss / (len(trainloader) * args.batch_size)
        #print(f"Energy: {energy_train}, Accuracy: {accuracy_train}, Loss: {loss_train}")
        writer.add_scalar(f"loss", loss_train, epoch + args.epochs*2)
        writer.add_scalar(f"energy", energy_train, epoch + args.epochs*2)
        writer.add_scalar(f"accuracy", accuracy_train, epoch + args.epochs*2)

        running_true_positive = 0.
        for i, (x, labels) in enumerate(tqdm(testloader)):
            x, labels = x.view(-1, 784).to(device), labels.to(device)
            h1, h2, y = states_test[i]

            # Free phase
            for j in range(100):
                dh1 = d_rho(h1) * (x @ W1 + h2 @ W2.t() + b1) - h1
                dh2 = d_rho(h2) * (h1 @ W2 + y @ W3.t() + b2) - h2
                dy = d_rho(y) * (h2 @ W3 + b3) - y

                h1 = rho(h1 + epsilon * dh1)
                h2 = rho(h2 + epsilon * dh2)
                y = rho(y + epsilon * dy)

            h1_free, h2_free, y_free = h1, h2, y
            states_test[i] = h1_free, h2_free, y_free

            running_true_positive += (y_free.argmax(1) == labels).sum().item()
        accuracy_test = running_true_positive / (len(testloader) * args.batch_size)
        writer.add_scalar(f"accuracy_test", accuracy_test, epoch + args.epochs*2)
        print(f"Energy: {energy_train}, Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}, Loss: {loss_train}")

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
        shuffle=False,
        num_workers=2)
    testloader = th.utils.data.DataLoader(
        mnist(root="./data", train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return trainloader, testloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--epochs-final", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)

    parser.add_argument("--fashion", action="store_true", default=False,
        help="use fashion mnist")
    parser.add_argument("--dir", default=utils.timestamp(),
        help="name of output log directory")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()
    device = th.device("cpu" if (not th.cuda.is_available() or args.no_cuda) else "cuda")
    device = th.device("cpu")
    main(args)
