'''NLP train data splitter.'''
from pathlib import Path

import torch.utils.data
from datasets import load_dataset
import numpy as np
import argparse

# CLI args
parser = argparse.ArgumentParser(description='NLP IMDB data splitting')
parser.add_argument('--n', '-n', type=int, help='the number of chunks to split into')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

# Path care
path = Path('./split_data/').expanduser()
path.mkdir(parents=True, exist_ok=True)
prefix = "imdb_split_part"


# Data
print('==> Preparing data..')
raw_datasets = load_dataset("imdb")
del raw_datasets["unsupervised"]
del raw_datasets["test"]
raw_datasets = raw_datasets.shuffle(seed=42)

trainset = raw_datasets["train"]

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
