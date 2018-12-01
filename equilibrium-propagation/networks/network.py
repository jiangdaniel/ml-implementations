#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from itertools import product

import numpy as np
import torch as th
import torch.nn as nn
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils


class Net(nn.Module):
    def __init__(self, n_layers, device="cpu"):
        assert n_layers > 0

        self.n_layers = n_layers

        a = np.sqrt(2. / (784. + 500.))
        self.weights = [th.randn(784, 500, device=device) * a]
        self.biases = [th.randn(500, device=device) * a]

        a = np.sqrt(2. / (500. + 500.))
        self.weights.extend([th.randn(500, 500, device=device) * a for _ in range(self.n_layers - 1)])
        self.biases.extend([th.randn(500, device=device) * a for _ in range(self.n_layers - 1)])

        a = np.sqrt(2. / (500. + 10.))
        self.weights.append(th.randn(500, 10, device=device) * a)
        self.biases.append(th.randn(10, device=device) * a)

        # Hyperparameters
        self.beta = 1.0
        self.epsilon = 0.5
        if n_layers == 1:
            self.n_iter_free = 20
            self.n_iter_clamped = 4
            self.alphas = [0.1, 0.05]
        elif n_layers == 2:
            self.n_iter_free = 100
            self.n_iter_clamped = 6
            self.alphas = [0.4, 0.1, 0.01]
        elif n_layers == 3:
            self.n_iter_free = 500
            self.n_iter_clamped = 8
            self.alphas = [0.128, 0.032, 0.008, 0.002]

    @staticmethod
    def rho(x):
        return x.clamp(0., 1.)

    @staticmethod
    def d_rho(x):
        return ((x >= 0.) * (x <= 1.)).float()

    def free_energy(self, x, units):
        """BROKEN: off by one on units"""
        raise NotImplementedError
        total = th.zeros(1)

        total += unit[0].pow(2).sum()
        total -= (self.weights[0] * (self.rho(x.t()) @ self.rho(units[0]))).sum()
        total -= (self.units[0] @ self.rho(self.biases[0])).sum() * 2.
        for i in range(1, len(self.weights)):
            total += unit[i].pow(2).sum()
            total -= (self.weights[i] * (self.rho(units[i-1].t()) @ self.rho(units[i]))).sum()
            total -= (self.units[i] @ self.rho(self.biases[i])).sum() * 2.

        return total.item() / 2.

    def fixed_points(self,  units, t):
        d_units = [None] * len(units)
        for _ in range(self.n_iter_free):
            for i in range(1, len(units) - 1):
                d_units[i] = self.d_rho(units[i]) * (units[i-1] @ self.weights[i-1] + units[i+1] @ self.weights[i].t() + self.biases[i-1]) - units[i]
            d_units[-1] = self.d_rho(units[-1]) * (units[-2] @ self.weights[-1] + self.biases[-1]) - units[-1]
            for i in range(1, len(units)):
                units[i] = self.rho(units[i] + self.epsilon * d_units[i])
        units_free = [u.clone() for u in units]

        for _ in range(self.n_iter_clamped):
            for i in range(1, len(units) - 1):
                d_units[i] = self.d_rho(units[i]) * (units[i-1] @ self.weights[i-1] + units[i+1] @ self.weights[i].t() + self.biases[i-1]) - units[i]
            d_units[-1] = self.d_rho(units[-1]) * (units[-2] @ self.weights[-1] + self.biases[-1]) - units[-1] + self.beta * (t - units[-1])
            for i in range(1, len(units)):
                units[i] = self.rho(units[i] + self.epsilon * d_units[i])

        return units_free, units

    def update(self, units_free, units_clamped):
        batch_size = units_free[-1].shape[0]
        for i in range(len(self.weights)):
            self.weights[i] += self.alphas[i] / self.beta * (self.rho(units_clamped[i].t()) @ self.rho(units_clamped[i+1]) - self.rho(units_free[i].t()) @ self.rho(units_free[i+1])) / batch_size
            self.biases[i] += self.alphas[i] / self.beta * (self.rho(units_clamped[i+1]) - self.rho(units_free[i+1])).mean(0)
