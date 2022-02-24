import os
import argparse
import warnings
import datasets

import torch
import flwr as fl

import pandas as pd
import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

from collections import OrderedDict

from utils import progress_bar

from pathlib import Path


# IF no tracking folder exists, create one automatically
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
else:
    if os.path.isfile('./checkpoint/loss_acc_tracking.txt'):
        os.remove('./checkpoint/loss_acc_tracking.txt')
    if os.path.isfile('./checkpoint/ckpt.pth'):
        os.remove('./checkpoint/ckpt.pth')

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint


# #############################################################################
# 1. Dataloader
# #############################################################################

def load_data(split_idx):
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)
    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    train_dd = raw_datasets["train"]

    if split_idx is not None:
        print('==> Training on a subset ', split_idx)
        path = Path('./split_data/').expanduser()
        prefix = "imdb_split_part"
        subset_idx = torch.load(path/(prefix+str(split_idx)+'.pt'))
        train_dl = torch.utils.data.DataLoader(subset_idx, shuffle=False)

        dat = []
        textgenerator = iter(train_dl)

        for i in range(len(subset_idx.indices)):
            try:
                etr = next(textgenerator)
                dat.append([etr['text'][0], np.array(etr['label'])[0]])
            except StopIteration:
                print(i)
        train_dd = datasets.arrow_dataset.Dataset.from_pandas(pd.DataFrame(dat, columns=['text', 'label']))

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train_dd = train_dd.map(tokenize_function, batched=True)
    tokenized_test_dd = raw_datasets["test"].map(tokenize_function, batched=True)

    tokenized_train_dd = tokenized_train_dd.remove_columns("text")
    tokenized_train_dd = tokenized_train_dd.rename_column("label", "labels")

    tokenized_test_dd = tokenized_test_dd.remove_columns("text")
    tokenized_test_dd = tokenized_test_dd.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainloader = torch.utils.data.DataLoader(
        tokenized_train_dd,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator
    )

    testloader = torch.utils.data.DataLoader(
        tokenized_test_dd,
        batch_size=16,
        collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, optimizer, trainloader, epochs, scheduler):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            print(batch_idx)
            targets=data['labels'].to(DEVICE)
            print(data)
            batch = {k: v.to(DEVICE) for k, v in data.items()}
            optimizer.zero_grad()
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        scheduler.step()

        print("train," + str(train_loss) + "," + str(100.*correct/total) +
                        "," + str(correct) + "," + str(total) + "\n")


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


def test_save(net, testloader, best_acc, epoch):

    metric = load_metric("accuracy")
    test_loss = 0
    net.eval()

    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        test_loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    test_loss /= len(testloader.dataset)
    test_accuracy = metric.compute()["accuracy"]

    if test_accuracy > best_acc:
        print('Best accuracy', test_accuracy)

    with open("./checkpoint/loss_acc_tracking.txt", "a") as track:
        track.write("test," + str(test_loss) + "," + str(test_accuracy) + "\n")

    return test_accuracy



def main():

    parser = argparse.ArgumentParser(description='PyTorch IMDB Training')
    parser.add_argument('--ip', type=str, help='Server ip address to use')
    parser.add_argument('--idx', type=int, help='index number to use')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    """Create model, load data, define Flower client, start Flower client."""

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    optimizer = AdamW(net.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    trainloader, testloader = load_data(args.idx)

    epochs_step = 1

    # Flower client
    class IMDBClient(fl.client.NumPyClient):

        epoch_counter = 0
        best_acc = 0.0

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, optimizer, trainloader, epochs_step, scheduler)
            self.epoch_counter = self.epoch_counter + epochs_step
            self.best_acc = test_save(net, testloader, self.best_acc, self.epoch_counter)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    client = IMDBClient()
    fl.client.start_numpy_client(args.ip, client=client)

    print("==> best accuracy:", client.best_acc)


if __name__ == "__main__":
    main()
