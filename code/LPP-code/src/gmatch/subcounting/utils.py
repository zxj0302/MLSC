import os
import networkx as nx
from subprocess import Popen, PIPE
from pathlib import Path
from shutil import copyfile

import numpy as np
import gmatch



def integer_graph(g):
    mapping =dict(zip(g.nodes(), range(g.number_of_nodes())))
    new_g = nx.convert_node_labels_to_integers(g)
    return new_g, mapping




# 5 conversions
def nx_to_submatch(graph, filename, u=None, to_integers=True):
    """A networkx object to a submatch format file.
    """
    
    if to_integers:
        graph, mapping = integer_graph(graph) # required, submatch format requires this.
        if u is not None:
            u = mapping[u]

    edges = graph.edges()
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    with open(filename, 'w') as f:
        f.write('t {} {}\n'.format(num_nodes, num_edges))
        if u is None:
            for i in range(num_nodes):
                f.write('v {} {} {}\n'.format(i, 0, graph.degree(i)))
        else:
            for i in range(num_nodes):
                f.write('v {} {} {}\n'.format(i, int(i==u), graph.degree(i)))
        for edge in edges:
            f.write('e {} {}\n'.format(edge[0], edge[1]))

def nx_to_edgefile(G, filename, to_integers=False):
    if to_integers:
        G = nx.convert_node_labels_to_integers(G)
    edges = list(G.edges())
    np.savetxt(filename, edges, fmt='%d', delimiter=' ')

def submatch_to_edges(filename):
    """
    A submatch file to an edge array.
    """
    edges = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == 'e':
            u, v = list(map(lambda x:int(x), line.split(' ')[1:] ))
            edges.append((u, v))
    return np.array(edges, dtype=int)

def submatch_to_nx(filename):
    """A submatch format file to a networkx object """
    G = nx.Graph()
    edges = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_nodes = int(lines[0].split()[1])
        for line in lines:
            if line[0] == 'e':
                u, v = list(map(lambda x:int(x), line.split(' ')[1:] ))
                edges.append((u, v))
    
    G.add_nodes_from(range(n_nodes)) # required
    G.add_edges_from(edges)
    # G = nx.convert_node_labels_to_integers(G)
    return G

def edges_to_submatch(file1, file2, u=None):
    """
    An edgelist file to a subgraph match format file.

    file1: input file with the edgelist format
    file2: output file with the SubgraphMatching format
    """
    G = nx.read_edgelist(file1, delimiter=' ', nodetype=int)
    nx_to_submatch(G, file2, u)

    # G = nx.convert_node_labels_to_integers(G) # relable nodes
    # num_nodes = G.number_of_nodes()
    # num_edges = G.number_of_edges()

    # with open(file2, 'w') as f:
    #     f.write('t {} {}\n'.format(num_nodes, num_edges))
    #     for i in range(num_nodes):
    #         f.write('v {} {} {}\n'.format(i, 0, G.degree(i)))
    #     for edge in G.edges:
    #         f.write('e {} {}\n'.format(edge[0], edge[1]))

def edges_to_nx(filename, to_integers=False):
    G = nx.read_edgelist(filename, delimiter=' ', nodetype=int)
    if to_integers:
        G = nx.convert_node_labels_to_integers(G)
    return G

 # 3 compound conversions
def to_submatch(G, filename, u=None):
    if isinstance(G, nx.Graph):
        nx_to_submatch(G, filename, u)
    elif isinstance(G, Path) or isinstance(G, str):
        G = Path(G)
        if str(G).endswith('.edges'):
            edges_to_submatch(G, filename, u)
        elif str(G).endswith('.graph'):
            assert u is None, 'If G is a .graph file, u should not be designated.'
            copyfile(G, filename)
        else: raise ValueError('G format error in function to_submatch.')


def to_nx(G, to_integers=False):
    if isinstance(G, nx.Graph):
        if to_integers:
            G = nx.convert_node_labels_to_integers(G)
        return G
    if isinstance(G, Path) or isinstance(G, str):
        G = Path(G)
        if str(G).endswith('.edges'):
            return edges_to_nx(G, to_integers)
        elif str(G).endswith('.graph'):
            return submatch_to_nx(G)
        else: raise ValueError('G format error in function to_nx.')

def to_edges(G):
    pass








def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments

def execute_binary(args):
    # import ipdb; ipdb.set_trace()
    process = Popen(' '.join(args), shell=True, stdout=PIPE, stderr=PIPE)
    (std_out, std_err) = process.communicate()
    process.wait()
    rc = process.returncode

    return rc, std_out, std_err

def subgraph_count(pattern_file, graph_file, binary_path):
    """
    graph_file: format of the subgraphmatching
    pattern_file: format of the subgraphmatching
    """
    execution_args = generate_args(binary_path, '-d', graph_file, '-q', pattern_file, '-filter', 'GQL',
                                    '-order', 'GQL', '-engine', 'LFTJ')
    rc, std_out, std_err = execute_binary(execution_args)
    if rc == 0:
        std_output_list = std_out.decode().split('\n')
        for line in std_output_list:
            if '#Embeddings' in line:
                embedding_num = int(line.split(':')[1].strip())
                return embedding_num
    else:
        print(f'Something wrong, rc: {rc}, std_err: {std_err}')
        # import ipdb; ipdb.set_trace()
        exit(-1)


BINARY_PATH = Path(os.path.dirname(os.path.abspath(__file__)))/'SubgraphMatching'/'build'/'matching'/'SubgraphMatching.out'
BINARY_PATH = str(BINARY_PATH)



def replace_line(filename, u):
    """Replace one line of a submatch format file with a new node label
    """
    pass


def subgraph_count_nx(pattern_graph, data_graph, u=None, v=None):
    """
    A wraper for `subgraph_count`, supporting directly receiving two networkx objects.
    Also support strings for graph files, which is faster and straightforward.

    Args:
        pattern_graph: a `submatch format` file or a `networkx object`, or an `.edges` file.
        data_graph: a `submatch format` file or a `networkx object`, or an `.edges` file.
        u: a root node in the pattern graph
        v: a root node in the data graph
    NOTE: Currently, u,v are useless. TODO: refine this.
    TODO: to support counting numbers under a given match pair {u <-> v}. (hint: setting node labels)
    """
    # generate temp files
    root_dir = gmatch.main.ROOT_DIR
    tmp_dir = root_dir/'data'/'tmp'
    if not tmp_dir.exists():
        os.makedirs(tmp_dir, exist_ok=True)


    if (isinstance(pattern_graph, Path) or isinstance(pattern_graph, str)) and str(pattern_graph).endswith('graph'):
        assert u is None, 'If pattern graph is already a .graph file, u should not be designated.'
        pattern_file = str(pattern_graph)
    else: # .edges or nx.Graph()
        pattern_file = str(tmp_dir/f'pattern_tmp_{os.getpid()}.graph')
        to_submatch(pattern_graph, pattern_file, u)
    
    if (isinstance(data_graph, Path) or isinstance(data_graph, str)) and str(data_graph).endswith('graph'):
        assert v is None, 'If data graph is already a .graph file, v should not be designated.'
        data_file = str(data_graph)
    else:
        data_file = str(tmp_dir/f'data_tmp_{os.getpid()}.graph')
        to_submatch(data_graph, data_file, v)

    
    count = subgraph_count(pattern_file, data_file, BINARY_PATH)

    if pattern_file.endswith(f'pattern_tmp_{os.getpid()}.graph'):
        try:
            os.remove(pattern_file)
        except:
            pass
    if data_file.endswith(f'data_tmp_{os.getpid()}.graph'):
        try:
            os.remove(data_file)
        except:
            pass

    return count
    
if __name__ == "__main__":
    import ipdb; ipdb.set_trace()
    print(BINARY_PATH)


    


