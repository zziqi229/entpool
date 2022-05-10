import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim import Adam

from data.dataloader import DataLoader

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None):
    val_losses, accs, durations = [], [], []
    kf = KFold(n_splits=10, shuffle=True, random_state=12345)
    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_index = train_index.astype(np.int64)
        test_index = test_index.astype(np.int64)
        train_dataset = dataset[train_index]
        test_dataset = dataset[test_index]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            test_loss = eval_loss(model, test_loader)
            train_acc = eval_acc(model, train_loader)
            test_acc = eval_acc(model, test_loader)
            accs.append(test_acc)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

    accs = np.array(accs).reshape(folds, epochs)
    accs_epoch = accs.mean(0)
    acc = accs[:, accs_epoch.argmax()]
    acc_mean, acc_std = acc.mean(), acc.std()
    print(f'Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}')

    return acc_mean, acc_std


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


criterion = torch.nn.CrossEntropyLoss()


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss = criterion(out, data.y.view(-1))
        total_loss += loss.item() * num_graphs(data)
    return total_loss / len(loader.dataset)
