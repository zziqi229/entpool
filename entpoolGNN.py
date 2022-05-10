from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

from torch_geometric.nn import GINConv, JumpingKnowledge, global_add_pool
from params import *


class Block(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mode='cat', num_layer=2):
        super().__init__()
        if (num_layer == 2):
            hidden_dim = output_dim
        self.num_layer = num_layer
        self.nns = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        self.nns.append(Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                                   Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))
        self.convs.append(GINConv(self.nns[-1]))
        for i in range(1, num_layer):
            self.nns.append(Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                                       Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU()))
            self.convs.append(GINConv(self.nns[-1]))

        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = Linear(hidden_dim * (num_layer - 1) + output_dim, output_dim)
        else:
            self.lin = Linear(output_dim, output_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        res = [x]
        for i in range(self.num_layer):
            res.append(self.convs[i](res[-1], edge_index))
        return self.lin(self.jump(res[1:]))


class entpoolGNN(torch.nn.Module):
    def __init__(self, dataset, hidden):
        super().__init__()
        self.k = dataset[0].layer_mask.shape[0] - 1
        self.embed_blocks = torch.nn.ModuleList()
        self.embed_blocks.append(Block(dataset.num_features, hidden, hidden, num_layer=3))
        for i in range(1, self.k):
            self.embed_blocks.append(Block(hidden, hidden, hidden, num_layer=1))

        self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(len(self.embed_blocks) * hidden, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.global_pooling = global_add_pool
        self.ent_pool = torch.nn.ModuleList()
        for i in range(self.k - 1):
            self.ent_pool.append(Block(hidden, hidden, hidden, num_layer=1))

    def reset_parameters(self):
        for embed_block in self.embed_blocks:
            embed_block.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def entpool(self, i, x, pool):
        x = self.ent_pool[i](x, pool)
        return x

    def forward(self, data):
        x, edge_index, layer_mask, batch = data.x, data.edge_index, data.layer_mask, data.batch
        pool = []
        for i in range(0, self.k):
            name = 'pool' + str(i)
            pool.append(data[name])

        xs = []
        for i, embed_block in enumerate(self.embed_blocks):
            mask = layer_mask[i].view(-1, 1)
            x = F.relu(embed_block(x, edge_index))
            xx = x * mask
            xs.append(self.global_pooling(xx, batch))
            # print(i, xs[-1].norm(1).item())
            if i < len(self.embed_blocks) - 1:
                x = self.entpool(i, xx, pool[i])

        # x = self.jump(xs)
        x = 0
        for i in range(len(xs)):
            x += xs[i] * config['alpha'][i]
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)
        return x

    def __repr__(self):
        return self.__class__.__name__
