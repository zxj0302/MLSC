import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from itertools import permutations, product
from collections import deque
from torch.utils.data import Dataset

from gmatch.preprocessing import makedirs
from gmatch.utils import load_pickle
from gmatch.utils import color_str

def node_filter_rooted(q, g, u, v):
    deg_u = q.degree(u)
    deg_v = g.degree(v)
    if deg_v < deg_u:
        return False
    else:
        return True

def node_filter(q, g, u):
    deg = q.degree(u)
    g_nodes = list(g.nodes())
    g_nodes_candidate = list(filter(lambda x: g.degree(x)>=deg, g_nodes)) # label filter

    return g_nodes_candidate


def split_layers(g, root):
    layer_nodes = []
    visited = {root}
    queue = deque([{root}, ])
    while queue:
        curr_layer_nodes = queue.popleft()
        layer_nodes.append(curr_layer_nodes)
        visited = visited.union(curr_layer_nodes)

        next_layer_nodes = set()
        for n in curr_layer_nodes:
            neighbors = g.neighbors(n)
            for ng in neighbors:
                if ng not in visited:
                    next_layer_nodes.add(ng)
        if len(next_layer_nodes) > 0:
            queue.append(next_layer_nodes)
    
    layer_nodes = list(map(lambda x: list(x), layer_nodes))
    return layer_nodes


def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

def perm_one_layer(layer_nodes):
    """
    All permutations for one layer's splitted nodes.
    Args:
        layer_nodes: splitted layer nodes, 
        supported format: e.g., [[1,2],[3,4],[5]], e.g., [[1,2], [3,4], 5]

    Return:
        return a list of all permutations, one perm should also be a list.
    """

    for i, item in enumerate(layer_nodes):
        if not isinstance(item, Iterable):
            layer_nodes[i] = [item]


    cluster_perms = list(permutations(layer_nodes))
    all_perms = []
    for clu_perm in cluster_perms:
        clu_to_perms = []
        for clu in clu_perm:
            inner_perms = list(permutations(clu))
            clu_to_perms.append(inner_perms)
        this_clu_perm_perms = list(product(*clu_to_perms))
        all_perms.extend(this_clu_perm_perms)
    
    for i, perm in enumerate(all_perms):
        all_perms[i] = list(flatten(perm))
    
    return all_perms

def perm_generator(g, root):
    """
    General idea: 
    1, cluster nodes in one layer, then permute nodes in each cluster independently.
    2, remember permutations of different clusters.
    Args:
        g: a networkx graph, with a root node

    Return: 
        All permutations.
        e.g.,
        ([0], [1, 2, 6], [3, 4], [5])
        ([0], [1, 2, 6], [4, 3], [5])
        ([0], [2, 1, 6], [3, 4], [5])
        ([0], [2, 1, 6], [4, 3], [5])
        ([0], [6, 1, 2], [3, 4], [5])
        ([0], [6, 1, 2], [4, 3], [5])
        ([0], [6, 2, 1], [3, 4], [5])
        ([0], [6, 2, 1], [4, 3], [5])
    """


    layer_nodes = split_layers(g, root)
    for idx, curr_layer_nodes in enumerate(layer_nodes):
        n_nodes = len(curr_layer_nodes)
        if n_nodes < 3:
            continue
        labels = dict(zip(curr_layer_nodes, range(n_nodes)))
        for i in range(0, n_nodes-2):
            for j in range(i+1, n_nodes-1):
                if bool( set(g.neighbors(curr_layer_nodes[i])).intersection(g.neighbors(curr_layer_nodes[j])) ):
                    labels[curr_layer_nodes[j]] = labels[curr_layer_nodes[i]]
        
        if len(set(labels.values())) == 1 or  len(set(labels.values())) == n_nodes:
            continue # combine all/split all <==> leave as it is.
        
        # otherwise, split into sub-lists
        inv_labels = {}
        for k, v in labels.items():
            inv_labels[v] = []
        for k, v in labels.items():
            inv_labels[v].append(k)
        layer_nodes[idx] = list(inv_labels.values())


    each_layers_perms = []
    for l_nodes in layer_nodes:
        layer_perms = perm_one_layer(l_nodes)
        each_layers_perms.append(layer_perms)
    
    full_perms = list(product(*each_layers_perms))
    return full_perms


def perm_to_matrix(P):
    """
    Convert a permutation list to a permutation matrix.
    E.g., ([0],[1,3],[2]) => [0, 1, 3, 2]
    => [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]]
    """
    P = list(flatten(P))
    P = np.array(P, dtype=int)
    N = len(P)
    pi = np.zeros((N, N), dtype=int)
    pi[np.arange(N), P] = 1
    return pi    


def collate_fn(samples_list):
    assert len(samples_list) > 0
    n = len(samples_list[0])
    new_list = [[] for i in range(n)]

    for sample in samples_list:
        for i, i_item in enumerate(sample):
            new_list[i].append(i_item)

    for i, items_list in enumerate(new_list):
        new_list[i] = torch.cat(items_list, axis=0)
    return new_list



class WholeGraphSampler():
    """
    For sampling whole graphs.
    Regard each whole_graph as a consecutive range of rooted_graphs in the train/val/test_split list.
    The sampling indexes are rooted_graph indexes of the LPPDataset.
    """
    def __init__(self, index_file, name,  batch_size=8, shuffle=True):
        assert name in ['train', 'val', 'test'], f'Wrong name {name} in WholeGraphSampler.'
        self.index_file = index_file
        self.index = load_pickle(index_file)
        self.name = name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.chunks_start = self.index[ f'{self.name}_chunks_start' ] # load the correct chunks_start list
        self.n_whole_graphs = len(self.chunks_start) - 1
        self.n_batches = self.n_whole_graphs // self.batch_size
        self._data_indexes = np.arange(self.n_whole_graphs) # these are `whole_graph` indexes.

    def __shuffle__(self):
        """
        Randomly shuffle the _data_indexes.
        """
        self._data_indexes = np.random.permutation(self.n_whole_graphs)
    
    def __iter__(self):
        if self.shuffle:
            self.__shuffle__()
        
        for i in range(self.n_batches):
            data_indexes = self._data_indexes[i*self.batch_size: (i+1)*self.batch_size]
            r_indexes = []
            for idx in data_indexes:
                start = self.chunks_start[idx]
                end = self.chunks_start[idx+1]
                r_indexes.extend( list(range(start, end)) ) # convert `whole_graph` indexes to `rooted_graph` indexes
            if len(r_indexes) == 0: # len(r_indexes) may = 0
                continue
                # import ipdb; ipdb.set_trace()
            yield r_indexes

    def __len__(self):
        return self.n_batches


def batch_filter(batch):
    """ Avoid CUDA out of memory error
    """
    lengths = batch[1]
    if max(lengths) >= 500:
        return True
    else:
        return False

class LPPDataset(Dataset):
    def __init__(self, dataset, pattern_name, name):
        from gmatch.main import ROOT_DIR
        self.dataset = dataset
        self.pattern_name = pattern_name
        self.name = name # 'train'/'val'/'test'
        self.processed_data_dir = ROOT_DIR/'data'/f'{dataset}_{pattern_name}'/'processed'
        self.processed_data = self.build_dataset()

        self.data_tensor = self.processed_data['data_tensor']
        self.start_indexs = self.processed_data['start_indexs']
        self.lengths = self.processed_data['lengths']
        self.label_tensor = self.processed_data['label_tensor']
        self.w_indexes = self.processed_data['w_indexes']

        if self.name == 'train':
            from gmatch.preprocessing import PrepareDataset
            preparer = PrepareDataset(self.dataset, self.pattern_name)
            label_dict = load_pickle(preparer.label_dict_fn)
            self.max_nodes = label_dict['rg_nodes_max']
        
        
    def build_dataset(self):
        makedirs(self.processed_data_dir)
        processed_file = self.processed_data_dir/f'{self.name}.pt'
        if not processed_file.exists():
            print(color_str(f'Processed {self.name}.pt of the {self.dataset} dataset not found. Processing...', 'yellow'))
            data = self.process(processed_file)
            print(color_str(f'Processing {self.name}.pt of the {self.dataset} finished.', 'yellow'))
            return data
        else:
            print(color_str(f'Processed {self.name}.pt of the {self.dataset} file found.', 'green'))
            return torch.load(processed_file)

    def process(self, processed_file):
        """
        Generate processed dataset file from {.graph files, a count label .dict file}.
        node_filter works here.

        Note:
        scatter_indexs: [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, ...], used for scatter sum index.
        start_index: [0, 3, 7, ...], used for __getitem__.
        """

        from gmatch.preprocessing import PrepareDataset
        preparer = PrepareDataset(self.dataset, self.pattern_name)
        graph_files, graphs, label_dict = preparer.load_dataset(self.name)

        # Vectorization
        all_vecs = []
        start_indexs = [0, ]
        lengths = []
        labels = []
        w_indexes = []
        rg_nodes_max = label_dict['rg_nodes_max']
        for i, (fn, g) in tqdm(enumerate(zip(graph_files, graphs)), total=len(graph_files), desc=f'Processing {self.name} set'):
            key = fn.name.rstrip('.graph')
            root = label_dict['roots'][key]
            count = label_dict['counts'][key]
            w_index = label_dict['w_index'][key]

            adj = nx.adjacency_matrix(g, nodelist=list(range(len(g))))
            adj = adj.toarray()
            
            vecs = []
            perms = perm_generator(g, root)
            counter = 0
            for perm in perms:
                perm_matrix = perm_to_matrix(perm)
                new_adj = perm_matrix @ adj @ perm_matrix.T
                vec = new_adj.reshape(1, -1)
                padded_length = rg_nodes_max*rg_nodes_max
                vec = np.pad(vec, ((0, 0), (0, padded_length-vec.shape[1])), mode='constant', constant_values=(-1,))
                vecs.append(vec)
                counter += 1
                if counter > 500:
                    break
                
            vecs = np.concatenate(vecs, axis=0)
            lengths.append( vecs.shape[0] )
            start_indexs.append( start_indexs[-1] + vecs.shape[0] )
            all_vecs.append(vecs)
            labels.append(count)
            w_indexes.append(w_index)

        # To Tensors and save.
        data_tensor = np.concatenate(all_vecs, axis=0)
        data_tensor = torch.Tensor(data_tensor)
        lengths = torch.LongTensor(lengths)
        labels = torch.Tensor(labels)
        w_indexes = torch.LongTensor(w_indexes)

        processed_tensors = {
            'data_tensor': data_tensor,
            'start_indexs': start_indexs,
            'lengths': lengths,
            'label_tensor': labels,
            'w_indexes': w_indexes,
        }
        
        torch.save(processed_tensors, processed_file)
        return processed_tensors

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, idx):
        start = self.start_indexs[idx]
        end = self.start_indexs[idx+1]
        vecs = self.data_tensor[start:end]
        length = self.lengths[idx:idx+1] 
        label = self.label_tensor[idx:idx+1]
        w_index = self.w_indexes[idx:idx+1]
        return vecs, length, label, w_index
