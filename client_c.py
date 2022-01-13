from collections import OrderedDict
import warnings

import flwr as fl
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models import *

from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils import progress_bar

import os
import argparse


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################

criterion = nn.CrossEntropyLoss()

def train(net, lr, trainloader, epochs):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    net.train()
    for _ in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(net(inputs), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        scheduler.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def test_save(net, testloader, best_acc, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return best_acc


def load_data(split_idx):
    """Load CIFAR-10 (training and test set)."""
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


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main():
    """Create model, load data, define Flower client, start Flower client."""

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--ip', type=str, help='Server ip address to use')
    parser.add_argument('--idx', type=int, help='index number to use')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))


    # Load model
    net = SimpleDLA()
    net = net.to(DEVICE)

    # Load data (CIFAR-10)
    trainloader, testloader, num_examples = load_data(args.idx)

    # Flower client
    class CifarClient(fl.client.NumPyClient):

        epoch_counter = 0
        best_acc = 0.0

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, args.lr, trainloader, epochs=5)
            self.epoch_counter = self.epoch_counter + 5
            self.best_acc = test_save(net, testloader, self.best_acc, self.epoch_counter)
            return self.get_parameters(), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

    client = CifarClient()

    # Start client
    fl.client.start_numpy_client(args.ip, client=client)

    print("==> best accuracy:", client.best_acc)

if __name__ == "__main__":
    main()
