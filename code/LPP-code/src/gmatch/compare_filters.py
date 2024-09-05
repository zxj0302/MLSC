
"""
Compare different filters.
"""

import argparse
import sys
import time
import numpy as np
from tqdm import tqdm
from gmatch.main import ROOT_DIR
from gmatch.subcounting.utils import edges_to_nx, submatch_to_nx
from gmatch.utils import compute_central_node


class Filter():
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_dir = ROOT_DIR/'data'/dataset
        self.graph_files = list(self.data_dir.glob('*.graph'))
        self.gs = []
        for i, g_f in tqdm(enumerate(self.graph_files), total=len(self.graph_files), leave=False, desc='load graphs'):
            g = submatch_to_nx(g_f)
            self.gs.append(g)

    def filter(self, q, r_q, filter_type='degree'):
        filter_rate = []
        start_time = time.time()

        for i, g in tqdm(enumerate(self.gs), total=len(self.gs)):
            n = g.number_of_nodes()
            if filter_type == 'degree':
                filtered = self.degree_filter(g, q, r_q)
            elif filter_type == 'neig_degree':
                filtered = self.neigbor_degree_filter(g, q, r_q)
            elif filter_type == 'pseudo_iso':
                filtered = self.gql_filter(g, q, r_q)
            else: 
                raise ValueError('filter_type error.')
            filter_rate.append( filtered / n )
        
        end_time = time.time()
        filter_rate = np.mean(filter_rate) # average filter rate
        run_time = (end_time - start_time) / len(self.gs) * 1000 # average run time (ms) per sample
        return filter_rate, run_time

    def degree_filter(self, g, q, r_q):
        filtered = 0
        for node in g.nodes():
            if g.degree(node) < q.degree(r_q):
                filtered += 1
        return filtered

    def neigbor_degree_filter(self, g, q, r_q):
        filtered = 0
        neig_max_degree = 0
        for node in q[r_q]:
            if q.degree(node) > neig_max_degree:
                neig_max_degree = q.degree(node)
        
        for node in g.nodes():
            if g.degree(node) < q.degree(r_q):
                filtered += 1
                continue
            neig_max_degree_this = 0
            for neig in g[node]:
                if g.degree(neig) > neig_max_degree_this:
                    neig_max_degree_this = g.degree(neig)
            if neig_max_degree_this < neig_max_degree:
                filtered += 1
        
        return filtered

    def gql_filter(self, g, q, r_q):
        """
        A pseudo-iso check algprithm is used.
        """
        C = {} # candidates
        C[r_q] = set( list( filter(lambda n: g.degree(n)>=q.degree(r_q), g.nodes()) ) )
        for neig in q[r_q]:
            C[neig] = set( list( filter(lambda n: g.degree(n)>=q.degree(neig), g.nodes()) ) )
        
        N_u = q[r_q]
        filtered = g.number_of_nodes() - len(C[r_q])
        for v in C[r_q]:
            N_v = g[v]
            
            all_u_prime_matched = True
            for u_prime in N_u:
                this_u_prime_matched = False
                for v_prime in N_v:
                    if v_prime in C[u_prime]:
                        this_u_prime_matched = True
                        break
                if not this_u_prime_matched:
                    all_u_prime_matched = False
                    break

            if not all_u_prime_matched:
                filtered += 1 # filter out v from C[r_q]
        
        return filtered


def parse_args(argstring=None):
    parser = argparse.ArgumentParser('Converter script.')
    parser.add_argument('--filter', '-f', type=str, default='degree', help='Filter type')
    try:
        if argstring is not None: args = parser.parse_args(argstring)
        else: args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

if __name__ == '__main__':
    args = parse_args()
    datasets = ['zinc',]
    filter_types = ['degree', ]

    patterns_dir = ROOT_DIR/'data'/'patterns'

    for dataset in datasets:
        if dataset == 'g8_graphs':
            patterns = ['p_htw3_3_3_17', 'p_htw3_3_3_16', 'p_htw3_4_5_3', 'p_htw3_5_5_19']
        else:
            patterns = ['p_htw3_3_3_17', 'p_htw3_3_3_16', 'p_htw3_5_5_19', 'p_htw3_5_5_20']
        

        for filter_type in filter_types:
            node_filter = Filter(dataset)
        
            for pattern in patterns:
                q_f = patterns_dir/f'{pattern}.edges'
                q = edges_to_nx(q_f)
                r_q, ecc = compute_central_node(q)
                filter_rate, run_time = node_filter.filter(q, r_q, filter_type=filter_type)
                print(f'Dataset: {dataset}, Pattern: {pattern}, Filter type: {filter_type}')
                print(f'Filter rate: {filter_rate:.8f}, run time: {run_time:.8f}ms')
            print('\n')






    
