from collections import defaultdict, Counter
from typing import Union

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from networkx.generators import directed
import torch
from torch.functional import Tensor
import torch.optim as optim
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
import networkx as nx
import numpy as np
import random
import scipy.stats as stats
from tqdm import tqdm
from typing import List
import itertools
import matplotlib.pyplot as plt


def sample_neigh(graphs, size):
    ps = np.array([len(g) for g in graphs], dtype=np.float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        # graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            # new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh


cached_masks = None


def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    # v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    # v = [np.sum(v) for mask in cached_masks]
    return v


def wl_hash(g: nx.Graph, dim=64, node_anchored=False):
    if nx.number_of_selfloops(g) != 0:
        print("error")
        raise ValueError
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=np.int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=np.int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]], axis=0))
            # newvecs[n] = np.sum(vecs[list(g.neighbors(n)) + [n]], axis=0)
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))


def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(
            target,
            k=max_size,
            progress_bar=len(targets) < 10,
            node_anchored=node_anchored,
        )
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size:
                total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(
            sorted(counts.items(), key=lambda x: len(x[1]), reverse=True)
        )[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out


def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    ps = np.arange(1.0, 0.0, -1.0 / (k + 1)) ** 1.5
    # ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac) else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts


def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    # Base case
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G, node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # Recursive step:
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [
            nbr
            for nbr in list(G[w].keys())
            if nbr > node_id and nbr not in sg and nbr not in old_v_ext
        ]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac) else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            # if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps, node_anchored)
        sg.remove(w)


def gen_baseline_queries_mfinder(
    queries, targets, n_samples=10000, node_anchored=False
):
    sizes = Counter([len(g) for g in queries])
    # sizes = {}
    # for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)
        # bads, t = 0, 0
        # for ka, nas in counts.items():
        #    for kb, nbs in counts.items():
        #        if ka != kb:
        #            for a in nas:
        #                for b in nbs:
        #                    if nx.is_isomorphic(a, b):
        #                        bads += 1
        #                        print("bad", bads, t)
        #                    t += 1

        for _, neighs in list(
            sorted(counts.items(), key=lambda x: len(x[1]), reverse=True)
        )[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out


device_cache = None


def get_device():
    global device_cache
    if device_cache is None:
        device_cache = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # device_cache = torch.device("cpu")
    return device_cache


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == "adam":
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay
        )
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == "none":
        return None, optimizer
    elif args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate
        )
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart
        )
    return scheduler, optimizer


def relabel_dsgraph(data: DSGraph):
    """
    apply transform for each graph
    """
    x = data.node_feature
    edge_index = data.edge_index
    has_y = False
    if hasattr(data, "node_label"):
        if data.node_label != None:
            node_label = data.node_label

    map = torch.randperm(x.shape[0]).to(x.device)

    data.node_feature = x[map, :]  # relabel x
    data.edge_index = map[edge_index]  # relabel edge_index
    if has_y:
        data.node_label = node_label[map, :]


def add_node_feat_to_networkx(
    graph: nx.Graph, node_feats: List[torch.Tensor], node_feat_key: str = "feat"
):
    num_node = len(graph.nodes)
    num_feat = len(node_feats)

    num_results = num_feat**num_node

    output_graphs = [graph.copy() for i in range(num_results)]

    for i, feats in enumerate(itertools.product(node_feats, repeat=num_node)):
        for n, feat in enumerate(feats):
            output_graphs[i].nodes[n][node_feat_key] = feat

    return output_graphs
