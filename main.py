import torch
import numpy as np
import random

import argparse
from itertools import product
from entpoolGNN import entpoolGNN
from data.dataset import GNNDataset, myBatch

import wandb

wandb_log = False

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.optim as optim

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import time

from params import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = torch.nn.CrossEntropyLoss()

log_name = 'log/' + config['dataset'] + str(time.time()).split('.')[0] + '.log'


def log(s):
    with open(log_name, 'a') as fp:
        fp.write(s + '\n')


def train(model, optimizer, train_dataset):
    model.train()
    loss_sum = 0
    for t in range(config['iters_per_epoch']):
        selected_idx = np.random.permutation(len(train_dataset))[:config['batch_size']]
        data_list = [train_dataset[i] for i in selected_idx]
        # batch = torch_geometric.data.Batch.from_data_list(data_list)
        batch = myBatch(data_list)
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss * batch.num_graphs
    return loss_sum / len(train_dataset)


def test(model, dataset):
    model.eval()
    correct = 0
    loss_sum = 0
    batch_size = config['batch_size']
    cnt = int(np.ceil(len(dataset) / batch_size))
    for i in range(cnt):
        a = i * batch_size
        b = min((i + 1) * batch_size, len(dataset))
        data = myBatch(dataset[a:b])
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, data.y)
        loss_sum += loss * data.num_graphs
        pred = output.max(dim=1)[1]
        # print(pred)
        # print(data.y)
        correct += pred.view(-1).eq(data.y).sum().item()
    return correct / len(dataset), loss_sum / len(dataset)


if __name__ == '__main__':
    # wandb.init(project=config['dataset'], entity="zzq229", config=config)

    dataset = GNNDataset(name=config['dataset'], k=config['depth'], cleaned=True)

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch_geometric.seed.seed_everything(0)
    fold_idx = 0
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    labels = [graph.y for graph in dataset]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]

    # model = GINNet(input_dim=dataset.num_features,
    #                hidden_dim=config['hidden_dim'],
    #                num_layers=config['num_layers'],
    #                output_dim=dataset.num_classes,
    #                dropout=config['dropout'],
    #                ).to(device)
    model = entpoolGNN(dataset, config['hidden_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    accs = []
    log(str(config))
    mx_test_acc = 0
    for e in range(config['epoch']):
        scheduler.step()
        train_loss = train(model, optimizer, train_dataset)
        train_acc, t = test(model, train_dataset)
        test_acc, test_loss = test(model, test_dataset)
        accs.append(test_acc)
        mx_test_acc = max(mx_test_acc, test_acc)
        # wandb.log({"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc, "test_loss": test_loss})
        log(' Epoch: {:03d}, Train Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Acc: {:.7f}, Test Loss: {:.7f}   Max Test Acc: {:.7f}'.format(e, train_loss, train_acc, test_acc, test_loss, mx_test_acc))
        print(' Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Acc: {:.7f}, Test Loss: {:.7f}   Max Test Acc: {:.7f}'.format(e, train_loss, train_acc, test_acc, test_loss, mx_test_acc))
