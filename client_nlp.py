from collections import OrderedDict
import warnings

import flwr as fl
import torch

import random
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

import os
import argparse

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


def load_data():
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)

    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # random 100 samples
    population = random.sample(range(len(raw_datasets["train"])), 100)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(population)
    tokenized_datasets["test"] = tokenized_datasets["test"].select(population)

    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


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
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    trainloader, testloader = load_data()

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
            train(net, trainloader, epochs=epochs_step)
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
