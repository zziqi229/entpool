import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from data.graph2tree import graph2tree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, k, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'raw', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    G2T = graph2tree(dataset, k)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name, str(k))
    # dataset = TUDataset(path, name, cleaned=cleaned, pre_transform=graph2tree(dataset, k))
    dataset.data.edge_attr = None

    if dataset.data.x is None:
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
    if dataset.transform is None:
        dataset.transform = T.Compose(
            [G2T])
    else:
        dataset.transform = T.Compose(
            [dataset.transform, G2T])

    return dataset


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