#!/usr/bin/env python3

import os
import argparse

import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from tqdm import tqdm


class Network(object):
    def __init__(self):
        self.linear1 = Linear(64, 128)
        self.relu1 = ReLU()
        self.linear2 = Linear(128, 64)
        self.relu2 = ReLU()
        self.linear3 = Linear(64, 10)

    def forward(self, x):
        out = self.relu1(self.linear1(x))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def __call__(self, x):
        return self.forward(x)


class Linear(object):
    def __init__(self, input_size, output_size):
        self.W = np.zeros((input_size, output_size))
        self.cache = None
        self.reset_parameters()

    def forward(self, x):
        self.cache = x
        return x @ self.W

    def backward(self, grad):
        pass

    def reset_parameters(self):
        var = 1 / self.W.shape[0]
        self.W = np.random.normal(loc=0, scale=var, size=self.W.shape)

    def __call__(self, x):
        return self.forward(x)


class ReLU(object):
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.clip(x, a_min=0, a_max=None)

    def __call__(self, x):
        return self.forward(x)


def softmax(X):
    """https://deepnotes.io/softmax-crossentropy"""
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def cross_entropy(X, y):
    """https://deepnotes.io/softmax-crossentropy"""
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss


def main(args):
    data, target = datasets.load_digits(return_X_y=True)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data, target = data[indices], target[indices]

    splits = (int(0.7 * data.shape[0]), int(0.9 * data.shape[0]))
    scaler = preprocessing.StandardScaler().fit(data[:splits[0]])
    data = scaler.transform(data)
    train, val, test = zip(np.split(data, splits), np.split(target, splits))

    net = Network()

    for epoch in range(args.epochs):
        pred = net(train[0])
        loss = cross_entropy(pred, train[1])
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
