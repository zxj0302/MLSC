# get input and output paths from the command line
# translate the input data into the correct format
# save it at the path for experiments
# run the model on the data
# save the output at the path for experiments
# extract the runtime and sampling time from the output
# save the runtime and sampling time at the path for experiments
# remove the the generated data
# remove the the generated output
import argparse
import os
import subprocess
import shutil
import json
import torch
import numpy as np
import scipy.io as sio
import networkx as nx
from tqdm import tqdm
import logging
from typing import List, Any, Dict, Tuple
import itertools
import time
import resource
import csv
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXP_CONFIG = {
    "batch_size": [128],
    "target": list(range(29)),
    "model": ["NestedGIN_eff"],
    "h": [3],
    "lr": [1e-2],
    "layers": [3],
    "dataset": ["Set_1"]
}

target_diam = [2,1, 2,3,2,2,2,1, 2,3,4,2,3,2,3,3,2,3,2,2,2,2,2,2,2,2,2,2,1]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ESC-GNN experiments")
    parser.add_argument("input_folder", help="Path to the input folder containing dataset folders")
    parser.add_argument("output_folder", help="Path to the output folder")
    return parser.parse_args()

def create_graph_from_edge_index(data: Any) -> nx.Graph:
    G = nx.Graph()
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    if nx.number_of_selfloops(G) > 0:
        logger.warning("Graph has self-loops. Removing them...")
        G.remove_edges_from(nx.selfloop_edges(G))
    return G

def extract_counts(data: Any) -> np.ndarray:
    return data.gt_induced_le5[:, list(range(29))].numpy().astype(np.float32)

def convert_to_esc_format(dataset: Dict[str, List[Any]], output_file: str) -> bool:
    A_list, F_list = [], []
    index_arrays = {}

    start_idx = 0
    for split, data_list in dataset.items():
        end_idx = start_idx + len(data_list)
        index_arrays[f'{split}_idx'] = np.arange(start_idx, end_idx)
        start_idx = end_idx

        for data in tqdm(data_list, desc=f"Processing {split} graphs", unit="graph"):
            G = create_graph_from_edge_index(data)
            adj_matrix_np = nx.to_numpy_array(G)
            A_list.append(adj_matrix_np)
            
            cycle_counts = extract_counts(data)
            F_list.append(cycle_counts)

    A = np.array(A_list, dtype=object)
    F = np.array(F_list, dtype=object)

    graph_data = {
        'A': A,
        'F': F,
        **index_arrays
    }
    
    logger.info(f"Saving data to {output_file}")
    sio.savemat(output_file, graph_data)
    logger.info(f"Data saved to {output_file}")

    return True

def translate_input_data(input_folder: str):
    translated_datasets = []
    for dataset_name in sorted(os.listdir(input_folder)):
        dataset_folder = os.path.join(input_folder, dataset_name)
        if os.path.isdir(dataset_folder):
            dataset_file = os.path.join(dataset_folder, "dataset.pt")
            if os.path.exists(dataset_file):
                logger.info(f'Loading {dataset_file}...')
                dataset = torch.load(dataset_file)
                
                output_folder = f"/workspace/code/ESC-GNN/data/{dataset_name}/raw"
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, "data.mat")
                
                logger.info(f'Converting dataset {dataset_name} to ESC format...')
                completed = convert_to_esc_format(dataset, output_file)
                
                status = 'Successfully' if completed else 'With errors'
                logger.info(f'Conversion of {dataset_name} completed {status}.')
                
                if completed:
                    translated_datasets.append(dataset_name)
            else:
                logger.warning(f"dataset.pt not found in {dataset_folder}")
        else:
            logger.warning(f"{dataset_folder} is not a directory")
    return translated_datasets

def generate_esc_commands(config: Dict[str, Any], dataset: str) -> List[str]:
    base_command = "python run_graphcount.py"
    config_copy = config.copy()
    config_copy['dataset'] = [dataset]
    param_combinations = list(itertools.product(
        config_copy['batch_size'], config_copy['target'], config_copy['model'],
        config_copy['h'], config_copy['lr'], config_copy['layers'], config_copy['dataset']
    ))

    return [f"{base_command} --batch_size {p[0]} --target {p[1]} --model {p[2]} --h {target_diam[p[1]]} --lr {p[4]} --epochs 2000 --layers {p[5]} --dataset {p[6]}" for p in param_combinations]

def execute_command(full_command: str) -> Tuple[float, resource.struct_rusage, resource.struct_rusage, int]:
    start_time = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(full_command, shell=True) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    stdout, stderr = process.communicate()
    end_time = time.time()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    return end_time - start_time, usage_start, usage_end, process.returncode, stdout, stderr

def run_esc_gnn(dataset: str, output_folder: str):
    commands = generate_esc_commands(EXP_CONFIG, dataset)
    results = []
    output_folder = output_folder+f"/{dataset}/ESC-GNN"
    os.makedirs(output_folder, exist_ok=True)

    for cmd in tqdm(commands, desc=f"Running ESC-GNN on {dataset}", unit="command"):
        execution_time, usage_start, usage_end, return_code, stdout, stderr = execute_command(cmd)
        
        cpu_time = usage_end.ru_utime - usage_start.ru_utime
        memory_usage = usage_end.ru_maxrss - usage_start.ru_maxrss

        result = {
            "command": cmd,
            "execution_time": execution_time,
            "cpu_time": cpu_time,
            "memory_usage": memory_usage,
            "return_code": return_code,
            # "stdout": stdout.decode('utf-8'),
            # "stderr": stderr.decode('utf-8')
        }
    # Save results

    result_file = os.path.join(output_folder, f"{dataset}_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {result_file}")

def parse_results(res_folder, dataset_file, output_folder):
    full_counts = []
    times = []
    for target in range(29):
        json_file = f"{res_folder}/target_{target}.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
            full_counts.append(data["Prediction"])
            times.append(data["time_profile"])
    
    data = torch.load(dataset_file)
    test_data = data['test']
    slices = [i['num_nodes'] for i in test_data]
    # save full counts to csv, each column is a graph in full_counts
    with open(f'{output_folder}/prediction_node.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(*full_counts))
    
    # for each memebr of full_counts, sum the counts for each graph
    results = []
    for target in full_counts:
        graph_results = []
        start = 0
        for i in slices:
            graph_results.append(sum(target[start:start+i]))
            start += i
        results.append(graph_results)
    # do the division based on target
    for i, target in enumerate(results):
        if i in [0, 1]:
            results[i] = [x/3 for x in target]
        elif i in [2, 3, 4, 5, 6, 7]:
            results[i] = [x/4 for x in target]
        else:
            results[i] = [x/5 for x in target]
    # save the results to csv
    with open(f'{output_folder}/prediction_graph.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(*results))

    # add up the training, inference and sampling time
    added_times = {"training": 0, "inference": 0, "sampling": 0}
    for t in times:
        added_times["training"] += t["training"]
        added_times["inference"] += t["inference"]
        added_times["sampling"] += t["dataset_random_graph"]
    # save the runtime and sampling time at the path for experiments
    with open(f'{output_folder}/time.txt', 'w', newline='') as f:
        for key in added_times:
            f.write(f"{added_times[key]} ")
        f.write("\n")
        f.write("training, inference, sampling")



def main():
    args = parse_arguments()
    input_folder = args.input_folder
    output_folder = args.output_folder
    

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Translate input data into correct format and save it
    # translated_datasets = translate_input_data(input_folder)
    translated_datasets = ["Set_1"]

    # Run ESC-GNN on each translated dataset
    for dataset in translated_datasets[0:1]:
        run_esc_gnn(dataset, output_folder)

    # TODO: Extract runtime and sampling time from the output
    dataset = "Set_1"
    res_folder = f"{output_folder}/{dataset}/ESC-GNN/"
    dataset_file = f"/workspace/code/ESC-GNN/data/{dataset}/raw/dataset_compatible.pt"
    results = parse_results(res_folder, dataset_file, f"{output_folder}/{dataset}/ESC-GNN")


    
    # TODO: Save runtime and sampling time at the path for experiments
    # TODO: Remove the generated data
    # TODO: Remove the generated output

if __name__ == "__main__":
    main()