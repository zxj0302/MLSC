from datetime import datetime
import os
import torch
from tqdm.auto import tqdm
import networkx as nx
from joblib import Parallel, delayed
from torch_geometric.utils import to_networkx
import time


def get_ego_4h_graph(g_i):
    g = to_networkx(g_i[0], to_undirected=True)
    i = g_i[1]
    ego_4h_g = []
    for node in g.nodes:
        ego_full = nx.ego_graph(g, node, radius=4)
        ego_full = ego_full.subgraph([n for n in ego_full.nodes if n <= node])
        connected_components = list(nx.connected_components(ego_full))
        for cc in connected_components:
            if node in cc:
                ego_full = ego_full.subgraph(cc)
                break
        ego_4h_g.append(ego_full)

    G = nx.Graph()
    index_map = {}
    for index, ego in enumerate(ego_4h_g):
        local_map = {}
        for node in ego.nodes:
            local_map[node] = len(G.nodes) + len(local_map)
        index_map[index] = local_map[index]
        G.add_nodes_from(local_map.values())
        relabeled_edges = [(local_map[edge[0]], local_map[edge[1]]) for edge in ego.edges()]
        G.add_edges_from(relabeled_edges)

    with open(f'evoke/wrappers/edgelist_desco/rw_{i}.edgelist', 'w') as f:
        f.write(f'{len(G.nodes)} {len(G.edges)}\n')
        for edge in G.edges:
            f.write(f'{edge[0]} {edge[1]}\n')
    with open(f'evoke/wrappers/edgelist_desco/rw_{i}.index', 'w') as f:
        f.write(' '.join([str(x) for x in index_map.values()]))
    
class DatasetGenerator:
    def __init__(self):
        self.sample_path = 'RWDataset/samples/samples.pt'
        self.sample_gt_path = 'RWDataset/samples/samples_gt5.pt'

        self.evoke_wrappers_path = 'evoke/wrappers/'
        self.run_evoke_name = 'run_evoke.py'
        self.output_evoke_path = self.evoke_wrappers_path + 'output_evoke'
        self.edgelist_graph_path = self.evoke_wrappers_path + 'edgelist_graph'
        self.output_graph_path = self.evoke_wrappers_path + 'output_graph'
        self.edgelist_desco_path = self.evoke_wrappers_path + 'edgelist_desco'
        self.output_desco_path = self.evoke_wrappers_path + 'output_desco'

    def generate_desco_dataset(self, file=None):
        graphs = torch.load(self.sample_gt_path if file is None else file)
        print('GNN dataset loaded, total number of graphs: ', len(graphs))

        if os.path.exists(self.output_evoke_path):
            # rename the output_evoke folder to avoid overwriting, add timestamp in DDHHMMSS format
            timestamp = datetime.now().strftime("%d%H%M%S")
            os.rename(self.output_evoke_path, f"{self.output_evoke_path}_{timestamp}")
        os.mkdir(self.output_evoke_path)

        if os.path.exists(self.output_desco_path):
            # rename the output_desco folder to avoid overwriting, add timestamp in DDHHMMSS format
            timestamp = datetime.now().strftime("%d%H%M%S")
            os.rename(self.output_desco_path, f"{self.output_desco_path}_{timestamp}")
        os.mkdir(self.output_desco_path)

        _ = Parallel(n_jobs=min(os.cpu_count(), 32))(
            delayed(get_ego_4h_graph)(g_i) for g_i in tqdm([(g, i) for i, g in enumerate(graphs)], desc='Generating DeSCo ego networks')
        )

        print('DeSCo ego networks generated, edge list files generated and saved to files')

        os.system(f'cd {self.evoke_wrappers_path} && python {self.run_evoke_name} desco')
        for i, g in tqdm(enumerate(graphs), desc='Adding desco ground truth to graphs'):
            noninduced = torch.load(f'{self.output_desco_path}/noninduced_rw_{i}.pt')
            induced = torch.load(f'{self.output_desco_path}/induced_rw_{i}.pt')
            g.gt_noninduced_le5_desco = noninduced
            g.gt_induced_le5_desco = induced
        
        torch.save(graphs, 'RWDataset/samples/samples_desco.pt')
        print('DeSCo ground truth computed and saved')

if __name__ == "__main__":
    dg = DatasetGenerator()
    dg.generate_desco_dataset()