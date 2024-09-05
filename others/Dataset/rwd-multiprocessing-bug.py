import os
import torch
from tqdm.auto import tqdm
import networkx as nx
from multiprocessing import Pool, cpu_count
from torch_geometric.utils import to_networkx


def get_ego_4h_graph(g_i):
    try:
        g = to_networkx(g_i[0], to_undirected=True)
        i = g_i[1]
        ego_4h_g = []
        for node in g.nodes:
            ego_full = nx.ego_graph(g, node, radius=4)
            # delete nodes with index larger than node
            for n in list(ego_full.nodes):
                if n > node:
                    ego_full.remove_node(n)
            # get the connected components of the ego graph and select the one contains the node
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
            # write index_map.values as a list into f
            f.write(' '.join([str(x) for x in index_map.values()]))
    except Exception as e:
        print(f'Error in the {g_i[1]}th graph: {e}')
    return None
        
class DatasetGenerator:
    def __init__(self):
        self.ogb_path = 'download/OGB'
        self.planetoid_path = 'download/Planetoid'
        self.tudataset_path = 'download/TUDataset'

        self.sample_path = 'RWDataset/samples/samples.pt'
        self.sample_gt_path = 'RWDataset/samples/samples_gt5.pt'
        self.many_small_graph_path = 'RWDataset/many_small/graph.pt'
        self.many_small_desco_path = 'RWDataset/many_small/desco.pt'

        self.evoke_wrappers_path = 'evoke/wrappers/'
        self.run_evoke_name = 'run_evoke.py'
        self.output_evoke_path = self.evoke_wrappers_path + 'output_evoke'
        self.edgelist_graph_path = self.evoke_wrappers_path + 'edgelist_graph'
        self.output_graph_path = self.evoke_wrappers_path + 'output_graph'
        # Note: the ego network used in DeSCo contains only nodes having index < central node index
        self.edgelist_desco_path = self.evoke_wrappers_path + 'edgelist_desco'
        self.output_desco_path = self.evoke_wrappers_path + 'output_desco'

    def generate_desco_dataset(self, file=None):
        graphs = torch.load(self.sample_gt_path if file is None else file)
        print('GNN dataset loaded, total number of graphs: ', len(graphs))

        if os.path.exists(self.edgelist_desco_path):
            os.system(f'rm -rf {self.edgelist_desco_path}')
        os.mkdir(self.edgelist_desco_path)
        if os.path.exists(self.output_evoke_path):
            os.system(f'rm -rf {self.output_evoke_path}')
        os.mkdir(self.output_evoke_path)
        if os.path.exists(self.output_desco_path):
            os.system(f'rm -rf {self.output_desco_path}')
        os.mkdir(self.output_desco_path)

        # 3.1 for each graph in graphs, get desco-egonetwork around each node and combine them into a whole graph
        # 3.2 save the edgelists and indices to file

        with Pool(16) as p:
            _ = list(tqdm(p.imap_unordered(get_ego_4h_graph, [(g, i) for i, g in enumerate(graphs[21000:])]), total=len(graphs[21000:]), desc='Generating DeSCo ego networks'))
        print('DeSCo ego networks generated, edge list files generated and saved to files')

        # 3.3 compute the ground truth using 'run_ego.py' with EVOKE
        os.system(f'cd {self.evoke_wrappers_path} && python {self.run_evoke_name} desco')
        # read in the ground truth and add gt_induced_le5_desco & gt_noninduced_le5_desco to the graphs
        for i, g in tqdm(enumerate(graphs), desc='Adding desco ground truth to graphs'):
            noninduced = torch.load(f'{self.output_desco_path}/noninduced_rw_{i}.pt')
            induced = torch.load(f'{self.output_desco_path}/induced_rw_{i}.pt')
            g.gt_noninduced_le5_desco = noninduced
            g.gt_induced_le5_desco = induced
        # save the graphs to file
        torch.save(graphs, 'RWDataset/samples/samples_desco.pt')
        print('DeSCo ground truth computed and saved')

dg = DatasetGenerator()
dg.generate_desco_dataset()