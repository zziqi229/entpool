import torch
import numpy as np
import encoding_tree
import time
import copy


def find(x, parent, dep):
    if parent[x] == -1:
        dep[x] = 0
        return dep[x]
    if (dep[x] > 0):
        return dep[x]
    dep[x] = find(parent[x], parent, dep) + 1
    return dep[x]


def get_tree(input_):
    data, k = input_
    edges = data.edge_index.transpose(0, 1).numpy()
    G = encoding_tree.Graph(edges=edges, n=data.num_nodes)
    T = encoding_tree.Tree(G=G)
    parent = T.k_HCSE(k)
    parent = np.array(parent)
    dep = [-1] * parent.size
    dep = np.array(dep)
    for i in range(parent.size):
        dep[i] = find(i, parent, dep)
    return parent, dep


def graph2tree(input_):
    data, k = input_
    parent, dep = get_tree((data, k))
    dt = np.dtype([('dep', int), ('id', int)])
    node = [(-d, i) for i, d in enumerate(dep)]
    node = np.array(node, dtype=dt)
    node.sort(order='dep')

    data.num_edges = data.edge_index.shape[1]
    data.num_nodes = len(parent)
    data.x = torch.cat([data.x, torch.zeros(data.num_nodes - data.x.shape[0], data.x.shape[1])], dim=0)
    d = 0
    st, pn = 0, 0
    data.layer_mask = torch.zeros(k + 1, len(parent), dtype=torch.bool)
    for i in range(node.size):
        pn += 1
        if i + 1 == node.size or node[i][0] != node[i + 1][0]:
            data.layer_mask[d, st:pn] = True
            if i + 1 != node.size:
                t = torch.zeros(2, pn - st, dtype=torch.int64)
                for j in range(0, pn - st):
                    t[0, j], t[1, j] = j + st, parent[j + st]
                data['pool' + str(d)] = t
            d += 1
            st = pn
    layer_edge = [data.edge_index]
    # for i in range(k - 1):
    #     edge = copy.deepcopy(layer_edge[-1])
    #     edge = edge.reshape(-1)
    #     for j in range(edge.shape[0]):
    #         edge[j] = parent[edge[j]]
    #     edge = edge.reshape(2, -1)
    #     layer_edge.append(edge)
    data.edge_index = torch.cat(layer_edge, dim=1)

    return data
