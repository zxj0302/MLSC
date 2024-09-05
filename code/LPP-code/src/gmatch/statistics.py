"""
Show some statistics of datasets.
"""
from tqdm import tqdm
from gmatch.main import ROOT_DIR
from gmatch.subcounting.utils import submatch_to_nx
from gmatch.utils import Ploter
from gmatch.utils import load_pickle

class Statistic(object):
    def __init__(self, dataset, pattern=None):
        self.dataset = dataset
        self.pattern = pattern
        self.data_dir = ROOT_DIR/'data'/f"{self.dataset}_{self.pattern}"
    
    def dataset_statistics(self):
        """
        Output to console:
        Number of graphs
        Max |V|, Min |V|, Avg. |V|
        Max |E|, Min |E|, Avg. |E|
        """
        import numpy as np
        datasets = ['g8_graphs', 'ogbg_molhiv', 'qm9', 'zinc']
        for dataset in datasets:
            files = list((ROOT_DIR/'data'/dataset).glob('*.graph'))
            num_nodes = []
            num_edges = []
            
            for f in tqdm(files, total=len(files), leave=True):
                g = submatch_to_nx(f)
                num_nodes.append(g.number_of_nodes())
                num_edges.append(g.number_of_edges())
            max_n = np.max(num_nodes)
            min_n = np.min(num_nodes)
            avg_n = np.mean(num_nodes)
            max_e = np.max(num_edges)
            min_e = np.min(num_edges)
            avg_e = np.mean(num_edges)
            print(f'dataset: {dataset}')
            print(f'max |V|: {max_n}, min |V|: {min_n}, avg |V|: {avg_n}, max |E|: {max_e}, min |E|: {min_e}, avg |E|: {avg_e}')
        pass


    def show_whole_graph_nums(self):
        auxiliary_dict_fn = self.data_dir/'auxiliary'/'auxiliary.dict'
        auxiliary_dict = load_pickle(auxiliary_dict_fn)
        
        whole_root_start = auxiliary_dict['whole_root_start']
        print(f"{dataset} , {pattern}, {len(whole_root_start)-1}")

    def node_hist(self):
        """
        Number of nodes of rooted_graphs of a dataset
        """
        rg_dir = self.data_dir/"decomposed_graphs"
        rg_files = list(rg_dir.glob("*.graph"))

        num_nodes = []
        num_edges = []
        for rg_fn in tqdm(rg_files, total=len(rg_files), leave=True):
            g = submatch_to_nx(rg_fn)
            num_nodes.append(g.number_of_nodes())
            num_edges.append(g.number_of_edges())
        dic = {
            'Nodes': num_nodes,
            'Edges': num_edges
        }
        
        fn = ROOT_DIR/'figures'/f'{self.dataset}_{self.pattern}_rg_hist.pdf'
        Ploter.plot_hist(dic, figure_fn=fn, savefig=True)
    
    def subcounts_hist(self):
        import numpy as np
        label_dict_fn = self.data_dir/f"{self.pattern}.dict"
        auxiliary_dict_fn = self.data_dir/'auxiliary'/'auxiliary.dict'
        label_dict = load_pickle(label_dict_fn)
        auxiliary_dict = load_pickle(auxiliary_dict_fn)
        rg_subcounts = list(label_dict['counts'].values())
        
        whole_root_start = auxiliary_dict['whole_root_start']
        wg_subcounts = []
        for i in range(len(whole_root_start)-1):
            start = whole_root_start[i]
            end = whole_root_start[i+1]
            w_count = np.array(rg_subcounts[start:end]).sum()
            wg_subcounts.append(w_count)
        
        dic = {
            'Subgraph counts': wg_subcounts,
        }
        
        fn = ROOT_DIR/'figures'/f'{self.dataset}_{self.pattern}_wg_subcounts.pdf'
        Ploter.plot_hist(dic, figure_fn=fn, savefig=True, count_info=False)


def parse_args(argstring=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser('Converter script.')
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

    dataset = args.dataset
    pattern = args.pattern
    stat = Statistic(dataset, pattern)
    # stat.node_hist()
    # stat.subcounts_hist()
    # stat.show_whole_graph_nums()
    stat.dataset_statistics()