import numpy as np
import torch
import csv
from rdkit import Chem
import torch
from networkx import read_graph6
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader
import utils
import os
import random
import shutil
from itertools import repeat
# from k_gnn import GraphConv, DataLoader, avg_pool
# from k_gnn import TwoMalkin, ConnectedThreeMalkin
import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from torch_geometric.io import read_tu_data
import scipy.io as scio
import networkx as nx

# class dataset_random_graph(InMemoryDataset):
#     def __init__(self, url=None, dataname='random_graph', root='data', processed_name='processed', split='train',
#                  transform=None, pre_transform=None, pre_filter=None):
#         self.url = url
#         self.root = root
#         self.dataname = dataname
#         self.transform = transform
#         self.pre_filter = pre_filter
#         self.pre_transform = pre_transform
#         self.raw = os.path.join(root, dataname)
#         self.processed = os.path.join(root, dataname, processed_name)
#         super(dataset_random_graph, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
#                                             pre_filter=pre_filter)
#         split_id = 0 if split == 'train' else 1 if split == 'val' else 2
#         self.data, self.slices = torch.load(self.processed_paths[split_id])
#         self.y_dim = self.data.y.size(-1)
#         # self.e_dim = torch.max(self.data.edge_attr).item() + 1

#     @property
#     def raw_dir(self):
#         name = 'raw'
#         return os.path.join(self.root, self.dataname, name)

#     @property
#     def processed_dir(self):
#         return self.processed
#     @property
#     def raw_file_names(self):
#         names = ["data"]
#         return ['{}.mat'.format(name) for name in names]

#     @property
#     def processed_file_names(self):
#         return ['data_tr.pt', 'data_val.pt', 'data_te.pt']

#     def adj2data(self, A, y):
#         # x: (n, d), A: (e, n, n)
#         # begin, end = np.where(np.sum(A, axis=0) == 1.)
#         begin, end = np.where(A == 1.)
#         edge_index = torch.tensor(np.array([begin, end]))
#         # edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
#         # y = torch.tensor(np.concatenate((y[1], y[-1])))
#         # y = torch.tensor(y[-1])
#         # y = y.view([1, len(y)])

#         # sanity check
#         # assert np.min(begin) == 0
#         num_nodes = A.shape[0]
#         if y.ndim == 1:
#             y = y.reshape([1, -1])
#         return Data(x = torch.ones(num_nodes).view(num_nodes, 1).repeat(1, 10), edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

#     @staticmethod
#     def wrap2data(d):
#         # x: (n, d), A: (e, n, n)
#         x, A, y = d['x'], d['A'], d['y']
#         x = torch.tensor(x)
#         begin, end = np.where(np.sum(A, axis=0) == 1.)
#         edge_index = torch.tensor(np.array([begin, end]))
#         edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
#         y = torch.tensor(y[-1:])
#         return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

#     def process(self):
#         # process npy data into pyg.Data
#         print('Processing data from ' + self.raw_dir + '...')
#         raw_data = scio.loadmat(self.raw_paths[0])
#         if raw_data['F'].shape[0] == 1:
#             data_list_all = [[self.adj2data(raw_data['A'][0][i], raw_data['F'][0][i]) for i in idx]
#                              for idx in [raw_data['train_idx'][0], raw_data['val_idx'][0], raw_data['test_idx'][0]]]
#         else:
#             data_list_all = [[self.adj2data(A, y) for A, y in zip(raw_data['A'][0][idx][0], raw_data['F'][idx][0])]
#                         for idx in [raw_data['train_idx'], raw_data['val_idx'], raw_data['test_idx']]]
#         for save_path, data_list in zip(self.processed_paths, data_list_all):
#             print('pre-transforming for data at'+save_path)
#             if self.pre_filter is not None:
#                 data_list = [data for data in data_list if self.pre_filter(data)]
#             if self.pre_transform is not None:
#                 temp = []
#                 for i, data in enumerate(data_list):
#                     if i % 100 == 0:
#                         print('Pre-processing %d/%d' % (i, len(data_list)))
#                     temp.append(self.pre_transform(data))
#                 data_list = temp
#                 # data_list = [self.pre_transform(data) for data in data_list]
#             data, slices = self.collate(data_list)
#             torch.save((data, slices), save_path)



class dataset_random_graph(InMemoryDataset):
    def __init__(self, dataname, root='data', processed_name='processed', split='train', transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.processed = os.path.join(root, dataname, processed_name)
        super(dataset_random_graph, self).__init__(root=root, transform=transform, pre_transform=pre_transform,pre_filter=pre_filter)
        split_id = 0 if split == 'train' else 1 if split == 'val' else 2
        self.data, self.slices = torch.load(self.processed_paths[split_id])
        print('Data loaded from ' + self.processed_paths[split_id])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataname, 'raw')

    @property
    def processed_dir(self):
        return self.processed

    @property
    def raw_file_names(self):
        return ['dataset_compatible.pt']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        os.system(f'cp /workspace/input/{self.dataname}/dataset_compatible.pt {self.raw_dir}')

    def process(self):
        raw_data = torch.load(self.raw_paths[0])
        for save_path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            data_list = [Data(x = torch.ones(d['x'].shape[0]).view(d['x'].shape[0], 1).repeat(1, 10), edge_index=d['edge_index'], y=d['y'].float(), num_nodes=torch.tensor([d['x'].shape[0]])) for d in raw_data[split]]
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            print('pre-transforming for data at '+save_path)
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in tqdm(data_list, desc='Pre-processing', file=sys.stdout)]
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)
