# Note: this script should be in the escape directory
import re
import os
import sys
import torch

# get the sampled data path and output path from the command line
input_path = sys.argv[1] # MLSC/input/
output_path = sys.argv[2] # MLSC/output/

# for all the folders named Set_i in the sampled_data_path folder, do the following:
for i in range(1, len([s for s in os.listdir(input_path) if re.match(r"Set_\d+$", s)]) + 1):
    dataset_name = f'Set_{i}'
    output_folder = os.path.join(output_path, dataset_name, 'EVOKE')
    # convert it to the correct format
    sampled_data = torch.load(os.path.join(input_path, dataset_name, 'dataset.pt'))['test']
    for i, s in enumerate(sampled_data):
        edge_list = os.path.join('wrappers', 'data', dataset_name, f'{i}.edges')
        os.makedirs(os.path.dirname(edge_list), exist_ok=True)
        with open(edge_list, 'w') as f:
            f.write(f'{s.num_nodes} {s.num_edges//2}\n')
            for edge in s.edge_index.T:
                if edge[0] < edge[1]:
                    f.write(f'{edge[0]} {edge[1]}\n')
        
    os.system(f'cd wrappers && python run.py {dataset_name} {output_folder}')