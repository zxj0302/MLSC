# Note: this script should be in the escape directory
import re
import os
import sys
import torch
import json
import warnings
warnings.filterwarnings("ignore")


def load_config(config_file):
    """Load configuration from JSON file"""
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file '{config_file}': {e}")
        sys.exit(1)

def main():
    config_file = sys.argv[1]
    config = load_config(config_file)
    input_path = config['input_path']
    output_path = config['output_path']
    num_cpus = config.get('num_cpus', 8)
    datasets = config.get('datasets', [])
    print(f"Processing datasets: {datasets}")

    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        output_folder = os.path.join(output_path, dataset_name, 'MOTIVO')

        # Load and convert dataset
        dataset_path = os.path.join(input_path, dataset_name, 'dataset.pt')
        sampled_data = torch.load(dataset_path)['test']

        if os.path.exists(os.path.join('wrappers', 'data', dataset_name)):
            print(f"Removing existing folder: {os.path.join('wrappers', 'data', dataset_name)}")
            os.system(f'rm -rf {os.path.join("wrappers", "data", dataset_name)}')
        os.makedirs(os.path.join('wrappers', 'data', dataset_name), exist_ok=True)
        # Write edge list to file
        print(f"Writing edge list...")
        
        for i, s in enumerate(sampled_data):
            edge_list = os.path.join('wrappers', 'data', dataset_name, f'{i}.edges')
            with open(edge_list, 'w') as f:
                f.write(f'{s.num_nodes} {s.num_edges//2}\n')
                for edge in s.edge_index.T:
                    f.write(f'{edge[0]} {edge[1]}\n')
        
        os.system(f'cd wrappers && python run.py {dataset_name} {output_folder} {num_cpus}')


if __name__ == "__main__":
    main()