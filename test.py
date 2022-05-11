import torch
import sys
import pickle
import numpy as np
from multiprocessing import Pool
import random
import time

from data.dataset import GNNDataset
from data.dataloader import DataLoader
from data.graph2tree import get_tree
from data.dataset import getTUDataset
from main import myBatch

st1, st2 = 0, 0


def print_edge(s, t=0):
    for i, j in s.T:
        print(int(i.item()) - t, int(j.item()) - t)


def print_tree(data, k):
    for i in range(k):
        print_edge(data['pool' + str(i)])
    print('------')


def test():
    global st, mx
    dataset = GNNDataset(name='NCI1', k=2)
    print(len(dataset))
    d = dataset[30]
    for i in range(d.edge_index.shape[1]):
        print(d.edge_index[0, i].item(), d.edge_index[1, i].item())
    # print_tree(d, k=2)
    # print_edge(d.edge_index[:, :d.edge_index.shape[1] // 2])
    # loader = DataLoader(dataset, batch_size=64, shuffle=False)
    # t = 0
    # for data in loader:
    #     t = data
    #     break

    # t=myBatch(dataset[:64])
    #
    # print(t.y.shape[0], t.y.sum().item())
    # print(t)
    # print(t.ptr)
    # print(t.layer_mask)
    # print(t.layer_mask.sum().item())
    # print_edge(t.pool0, t.ptr[1].item())
    # print('---------------')
    # print_edge(t.pool1, t.ptr[1].item())
    # print('---------------')
    # print_edge(t.pool2, t.ptr[1].item())


def hhh(i):
    t = random.random()
    time.sleep(t)
    return i * 2, i + 10


def calc_Q(data):
    parent, dep = get_tree((data, 2))
    for a, fa in enumerate(parent):
        print(a, fa)
    deg = [0] * data.num_nodes
    inc = [0] * len(parent)
    totc = [0] * len(parent)
    m = data.edge_index.shape[1]
    for i, j in data.edge_index.T:
        u, v = i.item(), j.item()
        deg[u] += 1
        if parent[u] == parent[v]:
            inc[parent[u]] += 1
    for i, d in enumerate(dep):
        if (d != 2):
            continue
        totc[parent[i]] += deg[i]
    Q = 0.0
    for i, d in enumerate(dep):
        if (d != 1):
            continue
        t = inc[i] / (2 * m) - totc[i] / (2 * m) * totc[i] / (2 * m)
        print(i, inc[i], totc[i], t)
        Q += t
    print(Q)
    return Q


if __name__ == '__main__':
    test()
    # import numpy as np
    # from communities.algorithms import louvain_method
    # from communities.visualization import draw_communities
    # from communities.visualization import louvain_animation
    #
    # adj_matrix = np.array([[0, 1, 1, 0, 0, 0],
    #                        [1, 0, 1, 0, 0, 0],
    #                        [1, 1, 0, 1, 0, 0],
    #                        [0, 0, 1, 0, 1, 1],
    #                        [0, 0, 0, 1, 0, 1],
    #                        [0, 0, 0, 1, 1, 0]])
    # communities, frames = louvain_method(adj_matrix)
    # # louvain_animation(adj_matrix, frames)
    # print(communities)
    # draw_communities(adj_matrix, communities)

    # dataset = getTUDataset('IMDB-BINARY')
    # res = []
    # for i, data in enumerate(dataset[:1]):
    #     t = calc_Q(data)
    #     res.append(t)
    #     if (i > 100):
    #         break
    # res = np.array(res)
    # print('max=', res.max(), ' min=', res.min(), ' mean=', res.mean(), ' std=', res.std())
    # test()
    # dataset = datasets.MNIST(root='./', train=True,
    #                          transform=transforms.ToTensor(), download=True)
    # print(dataset[np.array([1, 2, 4])])
    pass
