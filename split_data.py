'''Train CIFAR10 with PyTorch.'''
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

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

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# Dividing the training data into num_clients, with each client having equal number of images
num_clients = 5
traindata_split = torch.utils.data.random_split(trainset,
                            [int(trainset.data.shape[0] / num_clients) for _ in range(num_clients)])

for i in range(len(traindata_split)):
    print("==> saving part " + str(i+1) + " out of", len(traindata_split))
    torch.save(traindata_split[i], path/(prefix+str(i)+'.pt'))

print("==> asserting part 0 correctness.")
check_tensor = torch.load(path/(prefix+str(0)+'.pt'))
dataset1 = torch.utils.data.dataset.Subset(trainset, check_tensor.indices)
dataset2 = torch.utils.data.dataset.Subset(trainset, traindata_split[0].indices)
if all(np.equal(dataset1.indices, dataset2.indices)):
    print("==> assert OK.")
else:
    print("==> assert is NOT OK.")
