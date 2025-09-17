import argparse
import os
import subprocess
import json
import torch
import csv
import logging
import itertools
import time
import sys
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import resource

# Hard-coded constants that should remain in the code
ESCAPE_TARGET_DIAM = [2, 1, 2, 3, 2, 2, 2, 1, 2, 3, 4, 2, 3, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
TARGET_DIAM = [2, 1, 3, 2, 2, 2, 2, 1, 4, 3, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run I2-GNN experiments")
    parser.add_argument("config", help="Path to configuration JSON file")
    return parser.parse_args()

def generate_i2_commands(dataset: str, config: Dict) -> List[str]:
    base_command = "python run_count.py"
    exp_config = config["exp_config"].copy()
    exp_config['dataset'] = [dataset]
    param_combinations = list(itertools.product(
        exp_config['batch_size'], exp_config['target'], [config["models"][0]],
        exp_config['h'], exp_config['lr'], exp_config['layers'], exp_config['dataset']
    ))

    batch_size_map = config["batch_size"]
    epochs = config["training"]["epochs"]

    return [
        f"{base_command} --batch_size {batch_size_map[p[2]][-TARGET_DIAM[p[1]]]} --target {p[1]} --model {p[2]} "
        f"--h {TARGET_DIAM[p[1]]} --lr {p[4]} --epoch {epochs} --dataset {p[6]}"
        for p in param_combinations
    ]

def execute_command(command: str) -> Tuple[float, float, int, int, str]:
    start_time = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    _, stderr = process.communicate()
    
    end_time = time.time()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    
    execution_time = end_time - start_time
    cpu_time = usage_end.ru_utime - usage_start.ru_utime
    memory_usage = usage_end.ru_maxrss - usage_start.ru_maxrss
    
    return execution_time, cpu_time, memory_usage, process.returncode, stderr.decode('utf-8')

def run_i2_gnn(dataset: str, output_folder: str, config: Dict) -> None:
    commands = generate_i2_commands(dataset, config)
    results = []
    model_output_folder = os.path.join(output_folder, "retrain", dataset, config["models"][0])
    os.makedirs(model_output_folder, exist_ok=True)
    result_file = os.path.join(model_output_folder, f"{dataset}_results.json")

    for cmd in tqdm(commands, desc=f"Running {config['models'][0]} on {dataset}", unit="command"):
        execution_time, cpu_time, memory_usage, return_code, std_err = execute_command(cmd)
        
        results.append({
            "command": cmd,
            "execution_time": execution_time,
            "cpu_time": cpu_time,
            "memory_usage": memory_usage,
            "return_code": return_code,
            "stderr": std_err
        })

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
    
    # Load configuration from the provided file path
    config = load_config(args.config)
    
    input_folder = config["paths"]["input_folder"]
    output_folder = config["paths"]["output_folder"]
    
    os.makedirs(output_folder, exist_ok=True)

    for dataset in config["datasets"]:
        for model in config["models"]:
            # Create a copy of exp_config and update the model
            exp_config = config["exp_config"].copy()
            
            # Create a modified config for this iteration
            modified_config = config.copy()
            modified_config["exp_config"] = exp_config
            modified_config["models"] = [model]
            
            try:
                run_i2_gnn(dataset, output_folder, modified_config)

                res_folder = os.path.join(output_folder, dataset, model)
                dataset_file = os.path.join(config["paths"]["dataset_base_path"], dataset, "raw/dataset_compatible.pt")
                try:
                    parse_results(res_folder, dataset_file, os.path.join(output_folder, dataset, model))
                except Exception as e:
                    logger.error(f"Error processing results for {dataset} with {model}: {e}")
            except Exception as e:
                logger.error(f"Error running experiment for {dataset} with {model}: {e}")

if __name__ == "__main__":
    main()