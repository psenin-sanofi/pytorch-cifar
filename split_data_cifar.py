'''Split the CIFAR dataset.'''
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training data splitter')
parser.add_argument('--n', '-n', type=int, help='the number of chunks to split into')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))


path = Path('./split_indices/').expanduser()
path.mkdir(parents=True, exist_ok=True)
prefix = "split_part"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# Dividing the training data into num_clients, with each client having equal number of images
num_clients = 3
if args.n:
    num_clients = args.n

part_num = int(trainset.data.shape[0] / num_clients)
last_n = trainset.data.shape[0] - (part_num*2)
split_n = [part_num] * (num_clients-1)
split_n.append(last_n)

traindata_split = torch.utils.data.random_split(trainset, split_n)

for i in range(len(traindata_split)):
    print("==> saving part " + str(i+1) + " out of", len(traindata_split))
    torch.save(traindata_split[i], path/(prefix+str(i)+'.pt'))

print("==> asserting part 0 correctness.")
check_tensor = torch.load(path/(prefix+str(0)+'.pt'))
if all(np.equal(check_tensor.indices, traindata_split[0].indices)):
    print("==> assert OK.")
else:
    print("==> assert is NOT OK.")
