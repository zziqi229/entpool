import os.path as osp
import os
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

from data.graph2tree import graph2tree
from multiprocessing import Pool
import numpy as np


def getTUDataset(name, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'raw', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset[0].x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    return dataset


class GNNDataset(Dataset):
    def __init__(self, name, k, cleaned=True):
        self.name = name
        self.k = k
        dataset_raw = getTUDataset(name)
        self.num_features = dataset_raw.num_features
        self.num_classes = dataset_raw.num_classes

        path_p = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'processed')
        if not os.path.isdir(path_p):
            os.makedirs(path_p)
        path_p = osp.join(path_p, name + '_' + str(k) + '.pickle')
        if osp.exists(path_p) and cleaned == False:
            print('load graph tree')
            with open(path_p, 'rb') as fp:
                self.dataset = pickle.load(fp)
        else:
            print('calc graph tree')
            pool = Pool()
            self.dataset = pool.map(graph2tree, [(data, self.k) for data in dataset_raw])
            with open(path_p, 'wb') as fp:
                pickle.dump(self.dataset, fp)

    def __len__(self, ):
        return len(self.dataset)

    def __getitem__(self, id):
        if (type(id) == np.ndarray):
            res = []
            for i in id:
                res.append(self.dataset[i])
            return res
        return self.dataset[id]


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class myBatch():
    def __init__(self, data_list):
        self.x = torch.concat([data.x for data in data_list], 0)
        self.y = torch.concat([data.y for data in data_list])
        self.num_graphs = len(data_list)

        self.ptr = [data.x.shape[0] for data in data_list]
        for i in range(1, len(self.ptr)):
            self.ptr[i] += self.ptr[i - 1]
        self.ptr = torch.tensor([0] + self.ptr)
        edge_index0 = torch.concat([data.edge_index[0] + self.ptr[i] for i, data in enumerate(data_list)])
        edge_index1 = torch.concat([data.edge_index[1] + self.ptr[i] for i, data in enumerate(data_list)])
        self.edge_index = torch.stack([edge_index0, edge_index1], 0)

        self.batch = torch.concat([torch.ones(data.x.shape[0], dtype=torch.int64) * i for i, data in enumerate(data_list)])

        self.layer_mask = torch.cat([data.layer_mask for data in data_list], dim=-1)

        for i in range(15):
            pool = 'pool' + str(i)
            if pool in data_list[0].keys:
                t = [data[pool] + self.ptr[i] for i, data in enumerate(data_list)]
                self[pool] = torch.cat(t, dim=-1)
            else:
                break

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        self.ptr = self.ptr.to(device)

        self.layer_mask = self.layer_mask.to(device)
        for i in range(15):
            pool = 'pool' + str(i)
            if pool in self.__dict__.keys():
                self[pool] = self[pool].to(device)
            else:
                break
        return self

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)
