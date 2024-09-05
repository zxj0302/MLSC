import torch
import numpy as np
import scipy.io as sio
import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_graph_from_edge_index(data: Any) -> nx.Graph:
    """Create a NetworkX graph from edge_index."""
    G = nx.Graph()
    # G.add_nodes_from(range(data.num_nodes))
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    # check if G has self-loops and print warning
    if nx.number_of_selfloops(G) > 0:
        logger.warning("Graph has self-loops. Removing them...")
        # remove self-loops from G
        G.remove_edges_from(nx.selfloop_edges(G))
    return G

def extract_counts(data: Any) -> np.ndarray:
    """Extract counts from gt_induced_le5 and cast them to float32."""
    return data.gt_induced_le5[:, [1,2,3,4,5,6,7,8]].numpy().astype(np.float32)

def parse_ratio(ratio_str: str) -> Tuple[int, int, int]:
    """Parse the input ratio string to integers."""
    try:
        train, val, test = map(int, ratio_str.split(':'))
        return train, val, test
    except ValueError:
        raise ValueError("Invalid ratio format. Use 'train:val:test', e.g., '4:1:1'")

def create_index_arrays(num_graphs: int, ratio: str, random_shuffle: bool = False) -> Dict[str, np.ndarray]:
    """Create index arrays for train, validation, and test sets."""
    train_ratio, val_ratio, test_ratio = parse_ratio(ratio)
    total_ratio = train_ratio + val_ratio + test_ratio
    
    train_size = int(num_graphs * train_ratio / total_ratio)
    val_size = int(num_graphs * val_ratio / total_ratio)
    
    all_indices = np.arange(num_graphs)
    if random_shuffle:
        np.random.shuffle(all_indices)

    return {
        'train_idx': all_indices[:train_size],
        'val_idx': all_indices[train_size:train_size+val_size],
        'test_idx': all_indices[train_size+val_size:]
    }

def convert_to_esc_format(dataset: List[Any], output_file: str, ratio: str, random_shuffle: bool) -> bool:
    """Convert dataset to ESC format and save as .mat file."""
    A_list, F_list = [], []

    for data in tqdm(dataset, desc="Processing graphs", unit="graph"):
        G = create_graph_from_edge_index(data)
        adj_matrix_np = nx.to_numpy_array(G)
        A_list.append(adj_matrix_np)
        
        cycle_counts = extract_counts(data)
        F_list.append(cycle_counts)

    A = np.array(A_list, dtype=object)
    F = np.array(F_list, dtype=object)

    index_arrays = create_index_arrays(len(dataset), ratio, random_shuffle)

    graph_data = {
        'A': A,
        'F': F,
        **index_arrays
    }
    
    logger.info(f"Saving data to {output_file}")
    sio.savemat(output_file, graph_data)
    logger.info(f"Data saved to {output_file}")

    return True

def convert_to_lpp_format(dataset: List[Any], output_file: str, ratio: str, random_shuffle: bool) -> bool:
    """Convert dataset to LPP format and save as .mat file."""
    # TODO: Implement LPP-specific conversion here
    logger.info(f"LPP conversion not yet implemented. Saving in ESC format to {output_file}")
    return convert_to_esc_format(dataset, output_file, ratio, random_shuffle)

def load_and_convert_dataset(input_file: str, output_file: str, ratio: str, random_shuffle: bool, format: str) -> None:
    """Load dataset and convert it to the specified format."""
    logger.info(f'Loading {input_file}...')
    dataset = torch.load(input_file)
    # unpack the dataset
    # dataset = dataset[0] + dataset[1] + dataset[2]

    # # sort the graphs by density
    # dataset = sorted(dataset, key=lambda data: data.num_edges / data.num_nodes)
    # # keep only the first 1/4 of the graphs
    # dataset = dataset[:100]
    # for data in dataset:
    #     print(data.num_edges / data.num_nodes)
    #     print(data.num_nodes)
    #     print(data.num_edges)
    #     G = create_graph_from_edge_index(data)
    #     print("max degree", max(dict(G.degree()).values()))
    #     print("min degree", min(dict(G.degree()).values()))
    
    if format == 'ESC':
        completed = convert_to_esc_format(dataset, output_file, ratio, random_shuffle)
    elif format == 'LPP':
        completed = convert_to_lpp_format(dataset, output_file, ratio, random_shuffle)
    else:
        raise ValueError(f"Unknown format: {format}")

    status = 'Successfully' if completed else 'With errors'
    logger.info(f'Conversion completed {status}.')

def main():
    parser = argparse.ArgumentParser(description="Dataset Conversion Tool")
    parser.add_argument('--input', type=str, required=True, help='Input filename (full path)')
    parser.add_argument('--output', type=str, required=True, help='Output filename (full path)')
    parser.add_argument('--ratio', type=str, default='4:1:1', help='Train:Val:Test ratio, e.g., 4:1:1')
    parser.add_argument('--random', action='store_true', help='Use random shuffling for index arrays')
    parser.add_argument('--format', type=str, choices=['ESC', 'LPP'], default='ESC', help='Output format')
    args = parser.parse_args()

    load_and_convert_dataset(args.input, args.output, args.ratio, args.random, args.format)

if __name__ == '__main__':
    main()