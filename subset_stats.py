import torch
import torchvision

import argparse

parser = argparse.ArgumentParser(description='counting class labels in a subset')
parser.add_argument('file')
args = parser.parse_args()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

subset_idx = torch.load(args.file)
dataset = torch.utils.data.dataset.Subset(trainset, subset_idx.indices)

buckets = [0]*10
for i in range(len(dataset.indices)):
    img, label = dataset[i]
    buckets[label] = buckets[label]+1

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

for i in range(10):
    print(classes[i], buckets[i])
