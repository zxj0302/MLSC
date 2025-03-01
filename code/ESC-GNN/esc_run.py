import argparse
import os
import subprocess
import json
import torch
import csv
import logging
import itertools
import time
import resource
from typing import List, Tuple, Dict
from tqdm import tqdm

# Constants
TARGET_DIAM = [2, 1,  3, 2, 2, 2, 2, 1,   4, 3, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
BATCH_SIZE = [128, 128, 128, 128]

# Datasets
DATASETS =  [f"Set_{i}" for i in range(1, 2)]

# Configuration
EXP_CONFIG = {
    "h": [3],
    "batch_size": [128],
    "target": [11],
    "dataset": "data_esc",
    "lr": [1e-2],
    "layers": [3],
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESC-GNN experiments")
    parser.add_argument("input_folder", help="Path to the input folder containing dataset folders")
    parser.add_argument("output_folder", help="Path to the output folder")
    return parser.parse_args()

def generate_esc_commands(dataset: str) -> List[str]:
    base_command = "python run_graphcount.py"
    config = EXP_CONFIG.copy()
    config['dataset'] = [dataset]
    param_combinations = list(itertools.product(
        config['batch_size'], config['target'], config['h'], 
        config['lr'], config['layers'], config['dataset']
    ))

    return [
        f"{base_command} --batch_size {BATCH_SIZE[-TARGET_DIAM[p[1]]]} --target {p[1]} "
        f"--model NestedGIN_eff --h {TARGET_DIAM[p[1]]} --lr {p[3]} --epochs 2000 "
        f"--layers {p[4]} --dataset {p[5]}"
        for p in param_combinations
    ]

def execute_command(command: str) -> Tuple[float, float, int, int, bytes]:
    start_time = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    
    end_time = time.time()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    
    execution_time = end_time - start_time
    cpu_time = usage_end.ru_utime - usage_start.ru_utime
    memory_usage = usage_end.ru_maxrss - usage_start.ru_maxrss
    
    return execution_time, cpu_time, memory_usage, process.returncode, stderr

def run_esc_gnn(dataset: str, output_folder: str) -> None:
    commands = generate_esc_commands(dataset)
    results = []
    model_output_folder = os.path.join(output_folder, "retrain", dataset, "ESC-GNN")
    os.makedirs(model_output_folder, exist_ok=True)

    for cmd in tqdm(commands, desc=f"Running ESC-GNN on {dataset}", unit="command"):
        execution_time, cpu_time, memory_usage, return_code, stderr = execute_command(cmd)
        
        results.append({
            "command": cmd,
            "execution_time": execution_time,
            "cpu_time": cpu_time,
            "memory_usage": memory_usage,
            "return_code": return_code,
            "stderr": stderr.decode('utf-8')
        })

        result_file = os.path.join(model_output_folder, f"{dataset}_results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info(f"Results saved to {result_file}")

def parse_results(res_folder: str, dataset_file: str, output_folder: str) -> None:
    full_counts = []
    times = []
    for target in range(29):
        json_file = os.path.join(res_folder, f"target_{target}.json")
        with open(json_file, 'r') as f:
            data = json.load(f)
            full_counts.append(data["Prediction"])
            times.append(data["time_profile"])
    
    data = torch.load(dataset_file)
    test_data = data['test']
    slices = [i['num_nodes'] for i in test_data]
    
    save_results(full_counts, output_folder, 'prediction_node.csv')
    graph_results = process_graph_results(full_counts, slices)
    save_results(graph_results, output_folder, 'prediction_graph.csv')
    save_time_results(times, output_folder)

def save_results(data: List[List[float]], output_folder: str, filename: str) -> None:
    with open(os.path.join(output_folder, filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(*data))

def process_graph_results(full_counts: List[List[float]], slices: List[int]) -> List[List[float]]:
    results = []
    for target in full_counts:
        graph_results = []
        start = 0
        for i in slices:
            graph_results.append(sum(target[start:start+i]))
            start += i
        results.append(graph_results)

    for i, target in enumerate(results):
        divisor = 3 if i in [0, 1] else 4 if i in range(2, 8) else 5
        results[i] = [x / divisor for x in target]
    
    return results

def save_time_results(times: List[Dict[str, float]], output_folder: str) -> None:
    added_times = {key: sum(t[key] for t in times) for key in ["training", "inference", "dataset_random_graph"]}
    
    with open(os.path.join(output_folder, 'time.txt'), 'w') as f:
        f.write(f"{added_times['training']} {added_times['inference']} {added_times['dataset_random_graph']}\n")
        f.write("training, inference, sampling")

def main() -> None:
    args = parse_arguments()
    input_folder = args.input_folder
    output_folder = args.output_folder
    
    os.makedirs(output_folder, exist_ok=True)

    for dataset in DATASETS:
        run_esc_gnn(dataset, output_folder)

        res_folder = os.path.join(output_folder, "retrain", dataset, "ESC-GNN")
        dataset_file = os.path.join("/workspace/code/ESC-GNN/data", dataset, "raw/dataset_compatible.pt")
        try:
            parse_results(res_folder, dataset_file, os.path.join(output_folder, dataset, "ESC-GNN"))
        except Exception as e:
            logger.error(f"Error processing results for {dataset} with ESC-GNN: {e}")

if __name__ == "__main__":
    main()