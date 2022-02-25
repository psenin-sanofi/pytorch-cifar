import os
import warnings

import torch
import flwr as fl

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

from collections import OrderedDict

from utils import progress_bar


# IF no tracking folder exists, create one automatically
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
else:
    if os.path.isfile('./checkpoint/single_loss_acc_tracking.txt'):
        os.remove('./checkpoint/single_loss_acc_tracking.txt')
    if os.path.isfile('./checkpoint/single_ckpt.pth'):
        os.remove('./checkpoint/single_ckpt.pth')

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
CHECKPOINT = "distilbert-base-uncased"  # transformer model checkpoint


# #############################################################################
# 1. Dataloader
# #############################################################################

def load_data():

    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)
    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(dd):
        return tokenizer(dd["text"], truncation=True)

    tokenized_train_dd = raw_datasets["train"].map(tokenize_function, batched=True)
    tokenized_train_dd = tokenized_train_dd.remove_columns("text")
    tokenized_train_dd = tokenized_train_dd.rename_column("label", "labels")

    tokenized_test_dd = tokenizer(raw_datasets["test"]["text"], truncation=True, padding=True)
    tokenized_test_dd = tokenized_test_dd.remove_columns("text")
    tokenized_test_dd = tokenized_test_dd.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainloader = torch.utils.data.DataLoader(
        tokenized_train_dd,
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator
    )

    testloader = torch.utils.data.DataLoader(
        tokenized_test_dd,
        batch_size=32,
        collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, optimizer, trainloader, epochs):

    net.train()

    for _ in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            targets = data['labels'].to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in data.items()}
            optimizer.zero_grad()
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        with open("./checkpoint/loss_acc_tracking.txt", "a") as track:
            track.write("train," + str(train_loss) + "," + str(100.*correct/total) +
                        "," + str(correct) + "," + str(total) + "\n")

def test_save(net, testloader, best_acc, epoch):

    metric = load_metric("accuracy")
    test_loss = 0
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():

        for batch_idx, data in enumerate(testloader):
            targets = data['labels'].to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in data.items()}
            outputs = net(**batch)
            loss = outputs.loss

            test_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with open("./checkpoint/loss_acc_tracking.txt", "a") as track:
        track.write("test," + str(test_loss) + "," + str(100.*correct/total) +
                    "," + str(correct) + "," + str(total) + "\n")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving... accuracy', acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return best_acc

def main():

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    optimizer = AdamW(net.parameters(), lr=5e-5)

    best_acc = 0.0

    trainloader, testloader = load_data()

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    for epoch in range(0, start_epoch+10):
        train(net, optimizer, trainloader, 1)
        best_acc = test_save(net, testloader, best_acc, epoch)

    print("==> best accuracy:", best_acc)


if __name__ == "__main__":
    main()
