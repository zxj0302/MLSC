"""
This module focuses on converting the dataset of gmatch to other dataset formats for other baselines.
"""
import argparse
import sys
import numpy as np
import dgl
import networkx as nx
from torch import Tensor
import dgl.data.utils as dgl_utils
from tqdm import tqdm
from gmatch.subcounting.utils import submatch_to_nx
from gmatch.utils import dump_pickle, load_pickle
from gmatch.preprocessing import makedirs
from gmatch.main import ROOT_DIR

class ConvertDataset(object):
    def __init__(self, dataset, pattern):
        self.dataset = dataset
        self.pattern = pattern
        self.data_dir = ROOT_DIR/'data'
        self.target_dir = self.data_dir/f"{self.dataset}"/'processed'/'auxiliary'
        self.whole_graph_label_dict_fn = self.target_dir/f"{self.pattern}.dict"

    def to_GNN(self):
        self.whole_graph_label_dict()
        print(f'[GNN] {self.dataset} {self.pattern} saved.')
        pass

    def whole_graph_label_dict(self):
        """
        Generate a whole_graph_label_dict file from the auxiliary.dict and pattern_name.dict file.
        Saved to dataset/'processed'/'auxiliary'/pattern.dict
        """
        new_dir =self.data_dir/f"{self.dataset}_{self.pattern}"
        auxiliary_dict_fn = new_dir/'auxiliary'/'auxiliary.dict'
        label_dict_fn = new_dir/f"{self.pattern}.dict"
        auxiliary_dict = load_pickle(auxiliary_dict_fn)
        label_dict = load_pickle(label_dict_fn)

        whole_graphs_list = auxiliary_dict['whole_graphs_list']
        rooted_graphs_list = auxiliary_dict['rooted_graphs_list']
        whole_root_start = auxiliary_dict['whole_root_start']
        counts = label_dict['counts']

        whole_graph_label_dict = {}
        for i, wg_fn in tqdm(enumerate(whole_graphs_list), total=len(whole_graphs_list)):
            y = 0
            rg_start = whole_root_start[i]
            rg_end = whole_root_start[i+1]
            for rg_i in range(rg_start, rg_end):
                rg_fn = rooted_graphs_list[rg_i]
                g_name = rg_fn.name.rstrip('.graph')
                rg_c = counts[g_name]
                y = y + rg_c
            wg_name = wg_fn.name.rstrip('.graph')
            whole_graph_label_dict[wg_name] = y
        
        
        makedirs(self.target_dir)
        dump_pickle(whole_graph_label_dict, self.whole_graph_label_dict_fn)

    def to_PPGN(self):
        """
        number of graphs
        N label
        i number_of_neighbors j1 j2 j3
        """
        self.ensure_whole_graph_label_dict_exists()
        whole_graph_label_dict = load_pickle(self.whole_graph_label_dict_fn)
        max_nodes = None

        glist = []
        labels = []
        for wg_name, subcount in tqdm(whole_graph_label_dict.items(), total=len(whole_graph_label_dict), desc='Converting to PPGN'):
            wg_fn = self.data_dir/f"{self.dataset}"/f"{wg_name}.graph"
            g_nx = submatch_to_nx(wg_fn)
            g_nx = nx.convert_node_labels_to_integers(g_nx) # https://discuss.dgl.ai/t/graph-loading-error-from-networkx/309/2
            glist.append(g_nx)
            labels.append(subcount)

        save_dir = ROOT_DIR.parent/'ProvablyPowerfulGraphNetworks_torch'/'data'/'benchmark_graphs'/f"{self.dataset}_{self.pattern}"
        save_fn = save_dir/f"{self.dataset}_{self.pattern}.txt"
        makedirs(save_dir)
        with open(save_fn, 'w') as f:
            f.write(f'{len(glist)}\n')
            for g_nx, subcount in zip(glist, labels):
                count = subcount
                f.write('{} {}\n'.format(g_nx.number_of_nodes(), count))
                for node in g_nx.nodes():
                    neigs = list(g_nx.neighbors(node))
                    numbers = [node, len(neigs), *neigs]
                    f.write(' '.join(  map(lambda x:str(x), numbers) )+'\n')
        print(f'[PPGN] {self.dataset} {self.pattern} saved.')

    def ensure_whole_graph_label_dict_exists(self):
        if not self.whole_graph_label_dict_fn.exists():
            print(f'whole_graph_label_dict file {self.whole_graph_label_dict_fn.name} does not exist. Processing...')
            self.whole_graph_label_dict()
        
    def to_LRP(self):
        """
        Format:
        1, a .bin file, consisting of a glist and a label dict.
        The two files are in one .bin file, loaded by the DGL.data.utils.load_graphs
            glist: a list of DGLGraph
            label: {'pattern': torch.Tensor([5000])}
           
        2, 3 index files, train.txt, test.txt, val.txt


        This model's training output is loss, not acc.
        """
        
        self.ensure_whole_graph_label_dict_exists()
        whole_graph_label_dict = load_pickle(self.whole_graph_label_dict_fn)

        save_dir = ROOT_DIR.parent/'GNN-Substructure-Counting'/'synthetic'/'data'
        file_bin = save_dir/f"{self.dataset}.bin"
        file_bin_exist = file_bin.exists()
        
        glist = []
        labels = {}
        labels[self.pattern] = []
        for wg_name, subcount in tqdm(whole_graph_label_dict.items(), total=len(whole_graph_label_dict), desc='Converting to LRP'):
            wg_fn = self.data_dir/f"{self.dataset}"/f"{wg_name}.graph"
            g_nx = submatch_to_nx(wg_fn)
            largest_cc = max(nx.connected_components(g_nx), key=len) # required, to avoid bugs in LRP (isolated nodes)
            g_nx = g_nx.subgraph(largest_cc).copy()
            if g_nx.number_of_nodes() == 1:
                continue # pass these graphs
            if not file_bin_exist:
                g_nx = nx.convert_node_labels_to_integers(g_nx) # https://discuss.dgl.ai/t/graph-loading-error-from-networkx/309/2
                g = dgl.DGLGraph(g_nx)
                glist.append(g)
            labels[self.pattern].append(subcount)
        labels[self.pattern] = Tensor(labels[self.pattern])

        if not file_bin.exists():
            dgl_utils.save_graphs(str(file_bin), glist, labels)
            N = len(glist)
            perms = np.random.permutation(N)
            train_idx = perms[:int(N*0.8)]
            val_idx = perms[int(N*0.8):int(N*0.9)]
            test_idx = perms[int(N*0.9):]

            file_train = save_dir/f"{self.dataset}_train.txt"
            file_val = save_dir/f"{self.dataset}_val.txt"
            file_test = save_dir/f"{self.dataset}_test.txt"
            np.savetxt(file_train, train_idx, fmt='%d')
            np.savetxt(file_val, val_idx, fmt='%d')
            np.savetxt(file_test, test_idx, fmt='%d')
            print(f'[LRP] {self.dataset} {self.pattern} saved.')
        else: # append new labels to the existed `all_labels` dict in the bin file
            glist, all_labels = dgl_utils.load_graphs(str(file_bin))
            all_labels[self.pattern] = labels[self.pattern]
            dgl_utils.save_graphs(str(file_bin), glist, all_labels)
            print(f'[LRP] {self.dataset} {self.pattern} saved (appended).')
        



def parse_args(argstring=None):
    parser = argparse.ArgumentParser('Converter script.')
    parser.add_argument('--model', '-m', type=str, default='GIN', help='Target model')
    parser.add_argument('--dataset', '-d', type=str, default='g8_graphs', help='Dataset name')
    parser.add_argument('--pattern', '-p', type=str, default='p_htw3_4_4_2', help='Pattern name')
    try:
        if argstring is not None: args = parser.parse_args(argstring)
        else: args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

if __name__ == '__main__':
    args = parse_args()

    converter = ConvertDataset(args.dataset, args.pattern)

    if args.model == 'GIN': # suitable for both GIN and GCN
        converter.to_GNN()
    elif args.model == 'PPGN': 
        converter.to_PPGN()
    elif args.model == 'LRP': 
        converter.to_LRP()
    else: 
        raise ValueError('Not supported')
