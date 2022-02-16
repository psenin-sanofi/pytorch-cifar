import os
import argparse
from collections import OrderedDict
import warnings
from pathlib import Path

import warnings
import flwr as fl
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    """Create model, load data, define Flower client, start Flower client."""

    parser = argparse.ArgumentParser(description='SkLearn CIFAR Training')
    parser.add_argument('--ip', type=str, help='Server ip address to use')
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    ##################################################################

    # Load CIFAR-10 (training and test set).
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    trainloader = None
    if split_idx is not None:
        print('==> Training on a subset ', split_idx)
        path = Path('./split_indices/').expanduser()
        prefix = "split_part"
        subset_idx = torch.load(path/(prefix+str(split_idx)+'.pt'))
        dataset = torch.utils.data.dataset.Subset(trainset, subset_idx.indices)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    else:
        print('==> Training on the full dataset')
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        print('==> training on ', len(trainloader.dataset), 'images')
        print('==> testing on ', len(testloader.dataset), 'images')

        num_examples = {"trainset": len(trainloader.dataset), "testset": len(testset)}
        return trainloader, testloader, num_examples

    def splitIndices(m, pCV):
        """ randomly shuffle a training set's indices, then split
        indices into training and cross validation sets. Pass in 'm'
        length of training set, and 'pCV', the percentage of the
        training set you would like to dedicate to cross validation."""
        # determine size of CV set.
        mCV = int(m*pCV)
        indices = np.random.permutation(m)
        return indices[mCV:], indices[:mCV]

    ##################################################################

    # Load data (CIFAR-10)
    trainloader, testloader, num_examples = load_data(args.idx)

    m = 50000  # total amount of data points
    pCV = 0.2  # % kept for CV

    trainIndices, testIndices = splitIndices(m, pCV)

    batchSize = 100
    tSampler = SubsetRandomSampler(trainIndices)

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class CIFARClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(args.ip, client=CIFARClient())
