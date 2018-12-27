#!/usr/bin/env python3

import argparse

import numpy as np
from skimage import io
import torch as th
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        self.mnist_size = 28 * 28
        self.encoder = Encoder(self.mnist_size, latent_size)
        self.decoder = Decoder(latent_size, self.mnist_size)

    def forward(self, x):
        x = x.view(-1, self.mnist_size)
        mean, logvar = self.encoder(x)
        latent = self.sample(mean, logvar)
        out = self.decoder(latent)
        return mean, logvar, out.view(out.shape[0], 28, 28)

    def sample(self, mean, logvar):
        std = th.exp(0.5 * logvar)
        z = th.randn_like(mean)
        return mean + z * std


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        hidden_size = (input_size + output_size * 2) // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return th.chunk(x, 2, dim=1)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        hidden_size = (input_size + output_size) // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).sigmoid()
        return x


def main(args):
    trainloader = get_trainloader(args.batch_size)
    vae = VAE(30).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for train_batch in tqdm(trainloader, desc=f"Epoch {epoch}", leave=False):
            train_batch[0] = train_batch[0].squeeze().to(device)
            mean, logvar, reconstruction = vae(train_batch[0])
            loss = F.binary_cross_entropy(reconstruction.view(-1, 784), train_batch[0].view(-1, 784), reduction='sum')
            #loss = (reconstruction - train_batch[0]).pow(2).sum()
            loss += -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    testloader = get_testloader(10)
    test_batch = iter(testloader).next()
    with th.no_grad():
        img = []
        test_batch[0] = test_batch[0].squeeze().to(device)
        _, _, reconstruction = vae(test_batch[0])
        for orig, recon in zip(test_batch[0], reconstruction):
            img.append(np.hstack((orig.detach().cpu().numpy(), recon.detach().cpu().numpy())))
        io.imsave("reconstructions.png", np.vstack(img))


def get_trainloader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),])
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


def get_testloader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),])
    testdataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform)
    testloader = th.utils.data.DataLoader(
        testdataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)
    return testloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()
    device = th.device("cpu" if (not th.cuda.is_available() or args.no_cuda) else "cuda")
    main(args)
