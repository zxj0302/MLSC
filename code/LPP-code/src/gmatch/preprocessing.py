
import argparse
import sys
import os
import networkx as nx
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from gmatch.subcounting.utils import nx_to_submatch, subgraph_count_nx, submatch_to_nx, integer_graph, to_nx
from gmatch.utils import dump_pickle, compute_central_node, load_pickle, noniso_graphs_to_nx
from gmatch.main import ROOT_DIR


def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)

def save_graphs(graphs, data_dir):
    for i, g in enumerate(graphs):
        file = data_dir/f'data_graph_{i}.graph'
        nx_to_submatch(g, file)

def statistics(graphs):
    mean_nodes = []
    mean_edges = []
    mean_degree = []
    for g in graphs:
        mean_nodes.append(g.number_of_nodes())
        mean_edges.append(g.number_of_edges())
        degree = np.mean(list( map(lambda x:x[1], g.degree() ) ) )
        mean_degree.append(degree)
    
    mean_nodes = np.mean(mean_nodes)
    mean_edges = np.mean(mean_edges)
    mean_degree = np.mean(mean_degree)
    stat = {'nodes': mean_nodes,
            'edges': mean_edges,
            'degree': mean_degree}
    return stat
def er_datagraphs(N, n, p):
    """Generate ER random graphs"""
    graphs = []
    for i in range(N):
        g = nx.erdos_renyi_graph(n, p) # Its nodes begin from 0.
        gc_nodes = max(nx.connected_components(g), key=len)
        gc = g.subgraph(gc_nodes).copy()
        gc = nx.convert_node_labels_to_integers(gc)
        graphs.append(gc)
    return graphs

def generate_random_datagraphs(N=5000,):
    set_random_seed()
    save_dir = ROOT_DIR/'data'/'synthetic'
    n_list = [10, 15, 20, 25, 30]
    p_list = [0.2, 0.3, 0.4, 0.5]
    params_list = []
    for n in n_list:
        for p in p_list:
            params_list.append((n, p))
    
    each_n = int(N/( len(params_list) ))
    total_number = 0
    all_graphs = []
    for params in params_list:
        graphs = er_datagraphs(each_n, *params)
        total_number = total_number + len(graphs)
        all_graphs = all_graphs + graphs
        
        stat = statistics(graphs)
        print(f'##### Params: {params} #####')
        print('Mean |V|: {}'.format(stat['nodes']))
        print('Mean |E|: {}'.format(stat['edges']))
        print('Mean V degree: {:.2f}'.format(stat['degree']))
        print('Num of graphs: {}'.format(len(graphs)))
    save_graphs(all_graphs, data_dir=save_dir)
    print(f'Total { total_number } graphs.')

    
class GraphsGenerater():
    def __init__(self) -> None:
        pass

    def generate_graphs(self, dataset, **kwargs):
        if dataset == 'random':
            self.random_graphs(**kwargs)
        elif dataset == 'g8':
            self.g8_graphs(**kwargs)

    @staticmethod
    def random_graphs(N=5000):
        set_random_seed()
        save_dir = ROOT_DIR/'data'/'synthetic'
        makedirs(save_dir)
        n_list = [10, 15, 20, 25, 30]
        p_list = [0.2, 0.3, 0.4, 0.5]
        params_list = []
        for n in n_list:
            for p in p_list:
                params_list.append((n, p))
        
        each_n = int(N/( len(params_list) ))
        total_number = 0
        all_graphs = []
        for params in params_list:
            graphs = er_datagraphs(each_n, *params)
            total_number = total_number + len(graphs)
            all_graphs = all_graphs + graphs
            
            stat = statistics(graphs)
            print(f'##### Params: {params} #####')
            print('Mean |V|: {}'.format(stat['nodes']))
            print('Mean |E|: {}'.format(stat['edges']))
            print('Mean V degree: {:.2f}'.format(stat['degree']))
            print('Num of graphs: {}'.format(len(graphs)))
        
        save_graphs(all_graphs, data_dir=save_dir)
        print(f'Total { total_number } graphs.')
    
    @staticmethod
    def g8_graphs(**kwargs):
        """
        http://users.cecs.anu.edu.au/~bdm/data/graphs.html
        """
        save_dir = ROOT_DIR/'data'/'g8_graphs'
        makedirs(save_dir)
        g8_file = ROOT_DIR/'data'/'all_noniso_graphs'/'graph8.all_graphs'
        g8_graphs = noniso_graphs_to_nx(g8_file)
        save_graphs(g8_graphs, data_dir=save_dir)
        print(f'Total { len(g8_graphs) } graphs.')
        pass
    
    def ogbg_molhiv(self):
        from ogb.graphproppred import PygGraphPropPredDataset
        data_dir = ROOT_DIR/'data'/'ogbg_molhiv'
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=data_dir) # download
        total_num = len(dataset) # 41127
        # choose 1/4
        choose_ratio = 0.25
        choose_num = int(total_num*choose_ratio)
        self.save_dataset(dataset, choose_num, data_dir, suffix='molhiv')

    def qm9(self):
        from torch_geometric.datasets import QM9
        data_dir = ROOT_DIR/'data'/'qm9'
        dataset = QM9(root=data_dir/'qm9')
        total_num = len(dataset)
        choose_ratio = 0.1
        choose_num = int(total_num*choose_ratio)
        self.save_dataset(dataset, choose_num, data_dir, suffix='qm9')
        pass
    
    def zinc(self):
        from torch_geometric.datasets import ZINC
        data_dir = ROOT_DIR/'data'/'zinc'
        dataset = ZINC(root=data_dir/'zinc')
        total_num = len(dataset)
        # choose 1/10
        choose_ratio = 0.1
        choose_num = int(total_num*choose_ratio)
        self.save_dataset(dataset, choose_num, data_dir, suffix='zinc')
    
    def save_dataset(self, dataset, choose_num, data_dir, suffix):
        for i in tqdm(range(choose_num), total=choose_num, desc=suffix):
            data = dataset[i]
            edges = data.edge_index
            edges_list = edges.numpy().T.tolist()
            g = nx.from_edgelist(edges_list, create_using=nx.Graph)
            fn = data_dir/f'{suffix}_{i}.graph'
            try:
                nx_to_submatch(g, fn, to_integers=True)
            except:
                # import ipdb; ipdb.set_trace()
                pass

def compute_subcounts(pattern_file, data_dir):
    """
    Input: a pattern, a dataset directory with many data_graphs
    """
    data_files = list(data_dir.glob('*.graph'))
    subcounts = {
        'pattern_file': str(pattern_file),
    }
    for data_file in tqdm(data_files, total=len(data_files), desc='compute_subcount'):
        count = subgraph_count_nx(pattern_file, data_file)
        subcounts[data_file.name] = count
    
    p_filename = str(pattern_file.name).split('.')[0] # e.g. p_htw3_4_5_3
    save_filename = p_filename + '.dict'
    save_filename = data_dir/save_filename
    dump_pickle(subcounts, save_filename)
    print(f'{p_filename} count saved.')


def split_train_val_test(itemlist, vt_ratio=0.3):
    """
    Split a list to train/val/test sets.

    Args:
        itemlist: a list of items for splitting.
        vt_ratio: ratio of val+test.
    """
    index = np.random.permutation(len(itemlist))
    train_end = int(len(itemlist) * (1-vt_ratio))
    val_end = train_end + int(len(itemlist) * vt_ratio/2)
    train_idx = index[:train_end]
    val_idx = index[train_end:val_end]
    test_idx = index[val_end:]

    train_list = []
    val_list = []
    test_list = []
    for idx in train_idx:
        train_list.append(itemlist[idx])
    for idx in val_idx:
        val_list.append(itemlist[idx])
    for idx in test_idx:
        test_list.append(itemlist[idx])
    
    return train_list, val_list, test_list
    

class Filter():
    def __init__(self, pattern, pattern_root):
        self.pattern = pattern
        self.pattern_root = pattern_root
        self.filters = [self.degree_filter, self.size_filter]
        pass

    def degree_filter(self, g, root):
        if g.degree(root) < self.pattern.degree(self.pattern_root):
            return True
        else: return False
    
    def size_filter(self, g, root):
        if g.number_of_nodes() < self.pattern.number_of_nodes() or g.number_of_edges() < self.pattern.number_of_edges():
            return True
        else: return False

    def __call__(self, g, root):
        for filter in self.filters:
            if filter(g, root):
                return True
        return False
        
class PrepareDataset():
    filenames = {
        'train': 'train_split.list',
        'val': 'val_split.list',
        'test': 'test_split.list',
        'w_sampler_index': 'WholeGraphSampler_index.dict',
        'auxiliary': 'auxiliary.dict',
    }

    def __init__(self, dataset, pattern_name):
        self.dataset = dataset
        self.pattern_name = pattern_name
        self.pattern_dir = ROOT_DIR/'data'/'patterns'
        files = list(self.pattern_dir.glob(f'*{pattern_name}*'))
        assert len(files) > 0, f'No pattern {pattern_name} in data/patterns dir.'
        self.pattern_file = files[0]
        
        
        self.data_dir = ROOT_DIR/'data'/dataset
        self.new_data_dir = ROOT_DIR/'data'/f'{dataset}_{self.pattern_name}'
        self.graph_files_dir = self.new_data_dir/'decomposed_graphs'
        self.auxiliary_dir = self.new_data_dir/'auxiliary'
        assert self.data_dir.exists(), f'Dataset {dataset} not exists in the data/ dir'
        makedirs(self.new_data_dir)
        makedirs(self.graph_files_dir)
        makedirs(self.auxiliary_dir)

        self.label_dict_fn = self.new_data_dir/f'{self.pattern_name}.dict'

    
    def load_dataset(self, name):
        rooted_graphs_fn = self.new_data_dir/self.filenames[name]
        graph_files = load_pickle(rooted_graphs_fn)

        label_dict = load_pickle(self.label_dict_fn)

        graphs = [submatch_to_nx(g_file) for g_file in graph_files]  
        return graph_files, graphs, label_dict

    def decompose_graphs(self, radius, filter):
        whole_graphs_list = []
        rooted_graphs_list = []
        rooted_graphs_root_list = []
        whole_root_start = [0, ]
        root_whole_identifier = []

        whole_graphs_list = list(self.data_dir.glob('*.graph'))
        for i_w, file in tqdm(enumerate(whole_graphs_list), total=len(whole_graphs_list), desc='Phase 1'):
            g = submatch_to_nx(file)
            r_count = 0
            for i, n in enumerate(g.nodes()):
                g_name = file.name.rstrip('.graph') + f'_{i}'
                g_sub_fn = self.graph_files_dir/f'{g_name}.graph'

                g_sub = nx.ego_graph(g, n, radius=radius)
                if filter(g_sub, n): continue
                g_sub, mapping = integer_graph(g_sub)
                n = mapping[n] # required
                r_count = r_count + 1 # may=0

                nx_to_submatch(g_sub, g_sub_fn, to_integers=False)
                rooted_graphs_list.append(g_sub_fn)
                rooted_graphs_root_list.append(n)
                root_whole_identifier.append(i_w)
            whole_root_start.append( whole_root_start[-1] + r_count )


        auxiliary_dict = {
            'whole_graphs_list': whole_graphs_list,
            'rooted_graphs_list': rooted_graphs_list,
            'rooted_graphs_root_list': rooted_graphs_root_list,
            'whole_root_start': whole_root_start,
            'root_whole_identifier': root_whole_identifier,
        }
        dump_pickle(auxiliary_dict, self.auxiliary_dir/self.filenames['auxiliary'])
        
        return auxiliary_dict

    def subcount_rooted_graphs(self, pattern, pattern_root, auxiliary_dict):
        label_dict = {}
        label_dict['pattern'] = self.pattern_name
        label_dict['roots'] = {}
        label_dict['counts'] = {}
        label_dict['w_index'] = {}
        label_dict['roots'][self.pattern_name] = pattern_root
        rg_nodes_num = []
        for rg_fn, rg_root, rg_w_id in tqdm(zip(auxiliary_dict['rooted_graphs_list'], auxiliary_dict['rooted_graphs_root_list'], \
                                                auxiliary_dict['root_whole_identifier']), total=len(auxiliary_dict['rooted_graphs_list']), \
                                                desc='Phase 2'):
            rg_nx = to_nx(rg_fn)
            rg_nodes_num.append(rg_nx.number_of_nodes())
            g_name = rg_fn.name.rstrip('.graph')
            label_dict['counts'][g_name] = subgraph_count_nx(pattern, rg_nx, pattern_root, rg_root)
            label_dict['roots'][g_name] = rg_root
            label_dict['w_index'][g_name] = rg_w_id
        label_dict['rg_nodes_max'] = max(rg_nodes_num) # influence the vector padding length
        dump_pickle(label_dict, self.label_dict_fn)
    
    def split_train_test(self, auxiliary_dict=None):
        if auxiliary_dict is None: auxiliary_dict = load_pickle(self.auxiliary_dir/self.filenames['auxiliary'])
        whole_graphs_list = auxiliary_dict['whole_graphs_list']
        w_indexs = np.arange(len(whole_graphs_list))
        train_w_indexes, val_w_indexes, test_w_indexes = split_train_val_test(w_indexs, vt_ratio=0.3)
        
        train_rooted_graphs_list = []
        val_rooted_graphs_list = []
        test_rooted_graphs_list = []

        train_rooted_graphs_chunks_start = [0, ]
        val_rooted_graphs_chunks_start = [0, ]
        test_rooted_graphs_chunks_start = [0, ]

        whole_root_start = auxiliary_dict['whole_root_start']
        rooted_graphs_list = auxiliary_dict['rooted_graphs_list']
        for w_idx in train_w_indexes:
            rg_idx_start = whole_root_start[w_idx]
            rg_idx_end = whole_root_start[w_idx+1]
            rg_nums = rg_idx_end - rg_idx_start
            train_rooted_graphs_list.extend( rooted_graphs_list[rg_idx_start:rg_idx_end] )
            train_rooted_graphs_chunks_start.append( train_rooted_graphs_chunks_start[-1] + rg_nums )
        for w_idx in val_w_indexes:
            rg_idx_start = whole_root_start[w_idx]
            rg_idx_end = whole_root_start[w_idx+1]
            rg_nums = rg_idx_end - rg_idx_start
            val_rooted_graphs_list.extend( rooted_graphs_list[rg_idx_start:rg_idx_end] )
            val_rooted_graphs_chunks_start.append( val_rooted_graphs_chunks_start[-1] + rg_nums )
        for w_idx in test_w_indexes:
            rg_idx_start = whole_root_start[w_idx]
            rg_idx_end = whole_root_start[w_idx+1]
            rg_nums = rg_idx_end - rg_idx_start
            test_rooted_graphs_list.extend( rooted_graphs_list[rg_idx_start:rg_idx_end] )
            test_rooted_graphs_chunks_start.append( test_rooted_graphs_chunks_start[-1] + rg_nums )
            pass
        
        dump_pickle(train_rooted_graphs_list, self.new_data_dir/self.filenames['train'])
        dump_pickle(val_rooted_graphs_list, self.new_data_dir/self.filenames['val'])
        dump_pickle(test_rooted_graphs_list, self.new_data_dir/self.filenames['test'])

        # WholeGraphSampler index file
        index_file_fn = self.new_data_dir/self.filenames['w_sampler_index']
        chunks_start_dict = {
            'train_chunks_start': train_rooted_graphs_chunks_start,
            'val_chunks_start': val_rooted_graphs_chunks_start,
            'test_chunks_start': test_rooted_graphs_chunks_start,
        }
        dump_pickle(chunks_start_dict, index_file_fn)


    def prepare_dataset(self):
        pattern_file = self.pattern_file
        pattern = to_nx(pattern_file, to_integers=False)
        root_p, ecc = compute_central_node(pattern)
        pattern_filter = Filter(pattern, root_p)

        auxiliary_dict = self.decompose_graphs(radius=ecc, filter=pattern_filter)
        self.subcount_rooted_graphs(pattern, root_p, auxiliary_dict)
        self.split_train_test(auxiliary_dict)

def makedirs(dir):
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        os.makedirs(dir)
    return dir



def parse_args(argstring=None):
    parser = argparse.ArgumentParser('Preprocessing script.')
    parser.add_argument('--opt', '-o', type=str, default='', help='Operation options' )
    parser.add_argument('--dataset', '-d', type=str, default='ogbg_molhiv', help='Dataset name string' )
    parser.add_argument('--pattern_name', '-p', type=str, default='p_htw3_3_3_16', help='Pattern name string' )
    parser.add_argument('--confirm', default=False, action='store_true', help='Cancel confirmation in prepare ' )

    try:
        if argstring is not None:
            args = parser.parse_args(argstring)
        else:
            args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.opt == 'generate': # generate data graphs
        generator = GraphsGenerater()
        # if args.dataset == 'random':
        #     generate_random_datagraphs(5000)
        if args.dataset == 'g8_graphs':
            generator.g8_graphs()
        if args.dataset == 'ogbg_molhiv':
            generator.ogbg_molhiv()
        elif args.dataset == 'zinc':
            generator.zinc()
        elif args.dataset == 'qm9':
            generator.qm9()

    elif args.opt == 'process': # prepare a dataset/process data graphs
        if not args.confirm:
            key = input(f'Continue processing with dataset {args.dataset} pattern {args.pattern_name}?(y/n)')
            if key == 'y': pass
            else: sys.exit(0)
        prepare = PrepareDataset(args.dataset, args.pattern_name)
        prepare.prepare_dataset()
    else: pass