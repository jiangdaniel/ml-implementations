from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch as th
import torch.nn as nn
from torch.distributions import normal


class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        layers = [(784, 1200), (1200, 1200), (1200, 10)]
        self.weights_mus = nn.ParameterList([nn.Parameter(th.randn(*layer)) for layer in layers])
        self.weights_rhos = nn.ParameterList([nn.Parameter(th.randn(*layer) - 3.) for layer in layers])

        self.biases_mus = nn.ParameterList([nn.Parameter(th.randn(layer[1])) for layer in layers])
        self.biases_rhos = nn.ParameterList([nn.Parameter(th.randn(layer[1]) - 3.) for layer in layers])
        '''
        self.mus = nn.ParameterList([
            nn.Parameter(th.randn(784, 1200)), nn.Parameter(th.randn(1200, 1200)),
            nn.Parameter(th.randn(1200, 10))])
        self.rhos = nn.ParameterList([
            a
        '''

        self.z = normal.Normal(0, 1)
        self.prior_sigma = 1.

    def forward(self, x):
        weights, biases = self._sample_weights()
        for W, b in zip(weights[:-1], biases[:-1]):
            x = (x @ W + b).relu()
        x = (x @ weights[-1] + biases[-1]).log_softmax(dim=1)
        return x, weights, biases

    def _sample_weights(self):
        weights = []
        biases = []
        for W_mu, W_rho, b_mu, b_rho in zip(self.weights_mus, self.weights_rhos, self.biases_mus, self.biases_rhos):
            weights.append(W_mu + self.z.sample(W_mu.shape) * (1. + W_rho.exp()).log())
            biases.append(b_mu + self.z.sample(b_mu.shape) * (1. + b_rho).log())
        return weights, biases

    def log_likelihood_prior(self, weights, biases):
        total = th.tensor(0)
        for W, b in zip(weights, biases):
            total = total + self.z.log_prob(W / self.prior_sigma).sum()
            total = total + self.z.log_prob(b / self.prior_sigma).sum()
        return total

    def log_likelihood_posterior(self, weights, biases):
        total = th.tensor(0)
        for W, W_mu, W_rho, b, b_mu, b_rho in zip(weights, self.weights_mus, self.weights_rhos, biases, self.biases_mus, self.biases_rhos):
            total = total + self.z.log_prob((W - W_mu) / (1. + W_rho.exp()).log()).sum() \
                    + self.z.log_prob((b - b_mu) / (1. + b_rho.exp()).log()).sum()
        return total
