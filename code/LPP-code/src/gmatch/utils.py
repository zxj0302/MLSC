import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from gmatch.subcounting.utils import to_submatch, to_nx, subgraph_count, BINARY_PATH


def noniso_graphs_to_nx(filename):
    """
    Parsing the .all_graphs file in the data/all_noniso_graphs directory
    The input file format is from the website http://users.cecs.anu.edu.au/~bdm/data/graphs.html.

    Return:
        a list of nx objects
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    all_graphs = []
    N = len(lines)
    i = 0
    while i < N:
        if lines[i].startswith('Graph'):
            i = i + 1
            nn, ne = list(map(lambda x: int(x), lines[i].split() ))
            
            edges = []
            i = i + 1
            while i < N:
                if lines[i] != '\n':
                    es = list(map(lambda x:[int(x.split()[0]), int(x.split()[1])],  lines[i].strip().split('  ')  )) 
                    edges.extend(es)
                    i = i + 1
                else:
                    break
            g = nx.Graph()
            g.add_nodes_from(range(nn))
            g.add_edges_from(edges)
            all_graphs.append(g)
        i = i + 1
    return all_graphs

def draw_graph_files(filenames):
    for filename in filenames:
        draw_filename = str(filename) + '.pdf'
        G = to_nx(filename)
        draw_nxg(G, draw_filename)


def draw_nxg(G, filename):
    """draw networkx graphs"""
    fig, axes = plt.subplots(1, 1, figsize=(6.4, 5.8))
    draw_params = {
        'node_color': np.array([[0, 0, 0]]),
        'node_size': 500,
        'with_labels': True,
        'font_color': 'white'
    }
    G = nx.convert_node_labels_to_integers(G)
    nx.draw_circular(G, ax=axes, **draw_params)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def draw_forbidden_minors():
    from gmatch.main import ROOT_DIR
    data_dir = ROOT_DIR/'data'/'forbidden_minors'
    minor_files = data_dir.glob('*.graph')
    draw_graph_files(minor_files)
    
def determine_isomorphic(G1, G2, u1=None, u2=None):
    """
    If u1/u2 is not None, write u1/u2 with label 1 into files. Other nodes are with label 0.
    Args:
        G1/G2: two graphs
        u1/u2: (if provided) root node of G1/G2

    isomorphic:
    |V1|==|V2|
    |E1|==|E2|
    subcount(G1, G2)>=1
    """
    from gmatch.main import TMP_DIR
    tmp_dir = TMP_DIR
    Gs = [G1, G2] # for precedence decisions
    Gs_files = [] # for submatch counts
    us = [u1, u2]
    for i, G in enumerate(Gs): 
        file = tmp_dir/f'iso_g{i}_{os.getpid()}.graph' # pid is important, for multiprocessing running.
        to_submatch(G, file, us[i])
        Gs_files.append(str(file))

        g_nx = to_nx(G)
        Gs[i] = g_nx

    # precedence
    if Gs[0].number_of_nodes() != Gs[1].number_of_nodes():
        return False
    if Gs[0].number_of_edges() != Gs[1].number_of_edges():
        return False

    # count
    pattern_file, data_file = Gs_files
    count = subgraph_count(pattern_file, data_file, BINARY_PATH)

    # decision
    if count >= 1:
        return True
    else: return False

def dump_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def vertex_eccentricity(g):
    """
    Compute the eccentricity of all nodes.
    https://mathworld.wolfram.com/GraphEccentricity.html
    """
    eccentricity = nx.eccentricity(g) # dict: {vertex: ecc}
    return eccentricity

def compute_central_node(g):
    """ Return the central node, the eccentricity of the central node.
    """
    eccentricity = vertex_eccentricity(g)
    min_value = min(eccentricity.values())
    candidates = list( filter(lambda x: eccentricity[x]==min_value, eccentricity.keys()) )

    central_node = max(candidates, key=g.degree)
    return central_node, eccentricity[central_node]




###########
bcolors = {
    'header': '\033[95m',
    'blue': '\033[94m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'ENDC': '\033[0m',
    'bold': '\033[1m',
    'underline': '\033[4m' # not supported
}
    
def color_str(string, color='red'):
    color_suffix = bcolors.get(color, '')
    end = bcolors.get('ENDC')
    return color_suffix + string + end

def interrupt_wraper(func):
    def handle_interrupt(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except KeyboardInterrupt:
            print(color_str('Interrupt detected, exited.', 'yellow'))
            sys.exit(1)
    
    return handle_interrupt

def makedirs(dir):
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        os.makedirs(dir)
    return dir

def get_logger(logfile=None, logmode='a'):
    import logging
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        color_str('%(asctime)s: %(message)s', 'red'),
        datefmt='%Y-%m-%d %H:%M:%S')

    if logfile is not None:
        fh = logging.FileHandler(logfile, mode=logmode) # append mode
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_default_logger(log_file, args=None, sys_argv=None):
    parent_dir = log_file.parent
    makedirs(parent_dir)
    logger = get_logger(log_file)
    if sys_argv is not None:
        logger.info('[Command line]: python ' + ' '.join(sys_argv))
    if args is not None:
        logger.info('[Args parsed]')
        logger.info('-'*40)
        for key, value in vars(args).items():
            logger.info('{}={}'.format(key, value))
        logger.info('-'*40)
    
    return logger


class Ploter():
    save_fig = True
    figure_path = Path('./figures')
    suffix = '.pdf'
    width = 6.4
    height = 4.8
    def __init__(self, savefig=True, figure_path='./figures', suffix='.pdf', **kwargs):
        self.figure_path = Path(figure_path) 
        self.savefig = savefig
        self.suffix = suffix
        self.kwargs = kwargs

    @classmethod
    def plot_hist(cls, dict, bin_nums=50, savefig=False, count_info=True, figure_fn='fig',):
        num = len(dict)
        fig, axes = plt.subplots(1, num, figsize=(cls.width*num, cls.height))
        if num == 1: axes = [axes]
        for i, (key, y) in enumerate(dict.items()):
            counts, bins, patches = axes[i].hist(y, bins=bin_nums, label=key, color='black')
            axes[i]. set_xlabel(key)
            axes[i].set_ylabel('Values')
            axes[i].legend()

            # add auxiliary count info for each bin
            if count_info:
                bin_centers = 0.5 * np.diff(bins) + bins[:-1]
                for count, x in zip(counts, bin_centers):
                    axes[i].annotate(str(int(count)), xy=(x, 0), xycoords=('data', 'axes fraction'), \
                        xytext=(0, -18), textcoords='offset points', va='top', ha='center', fontsize=2)
                    percent = '{:.2f}'.format(100 * float(count) / counts.sum())
                    axes[i].annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'), \
                        xytext=(0, -32), textcoords='offset points', va='top', ha='center', fontsize=2)
            
        if savefig: 
            if not figure_fn.parent.exists():
                os.makedirs(figure_fn.parent)
            fig.savefig(figure_fn, bbox_inches='tight')
            print(f"{figure_fn.name} saved.")
        plt.close(fig)


colors = {
    'b': 'blue',
    'g': 'green',
    'r': 'red',
    'c': 'cyan',
    'm': 'magenta',
    'y': 'yellow',
    'k': 'black',
}
line_styles = {
    '--': 'dashed line style',
    '-.': 'dash-dot line style',
}
markers = {
    'o': 'circle marker',
    'v': 'triangle_down marker',
    '^': 'triangle_up marker',
    '<': 'triangle_left marker',
    '>': 'triangle_right marker',
    '1': 'tri_down marker',
    '2': 'tri_up marker',
    '3': 'tri_left marker',
    '4': 'tri_right marker',
    '8': 'octagon marker',
    's': 'square marker',
    'p': 'pentagon marker',
    'P': 'plus (filled) marker',
    '*': 'star marker',
    'h': 'hexagon1 marker',
    'H': 'hexagon2 marker',
    '+': 'plus marker',
    'x': 'x marker',
    'X': 'x (filled) marker',
    'D': 'diamond marker',
}

def fmt_iterator(colors, line_styles, markers):
    from itertools import cycle
    colors_iter = cycle(colors)
    line_styles_iter = cycle(line_styles)
    markers_iter = cycle(markers)
    while True:
        fmt = next(colors_iter)+next(line_styles_iter)+next(markers_iter)
        yield fmt
