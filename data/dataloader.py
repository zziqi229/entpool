from typing import Union, List

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, HeteroData, Dataset, Batch


class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        data_list = batch
        elem = data_list[0]
        batch = Batch.from_data_list(data_list, self.follow_batch,
                                     self.exclude_keys)
        # num_features = elem.x.shape[1]
        # t = [torch.cat([data.x, torch.zeros(batch.ptr[i + 1] - batch.ptr[i] - data.x.shape[0], num_features)], dim=0) for i, data in enumerate(data_list)]
        # batch.x = torch.cat(t, dim=0)
        batch.layer_mask = torch.cat([data.layer_mask for data in data_list], dim=-1)
        for i in range(15):
            pool = 'pool' + str(i)
            if pool in elem.keys:
                t = [data_list[i][pool] + batch.ptr[i] for i in range(batch.y.shape[0])]
                batch[pool] = torch.cat(t, dim=-1)
            else:
                break
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset: Union[Dataset, List[Data], List[HeteroData]],
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: List[str] = [],
            exclude_keys: List[str] = [],
            **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        exclude_keys += ['layer_mask']
        for i in range(15):
            exclude_keys.append('pool' + str(i))
        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=Collater(follow_batch,
                                             exclude_keys), **kwargs)
