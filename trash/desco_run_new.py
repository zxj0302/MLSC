# Note: this script should be in the same directory as the main.py
# TODO: print status messages to the console
import re
import os
import sys
import torch
import shutil
import subprocess
import pandas as pd
from datetime import datetime
from torch_geometric.data import InMemoryDataset, Data

column_exchange = [0, 1, 3, 2, 5, 4, 6, 7, 10, 9, 8, 12, 13, 11, 15, 14, 16, 18, 17, 20, 19, 22, 21, 23, 24, 25, 26, 27, 28]

def run_pretrained(output_folder, dataset_name, idx):
    output_pretrained = os.path.abspath(os.path.join(output_folder, str(idx)))
    if os.path.exists(output_pretrained):
        os.system(f'rm -rf {output_pretrained}/*')
    neigh_ckpt, gossip_ckpt = get_nei_gossip_ckpt(idx)
    command_pretrained = f'python main.py --neigh_checkpoint {neigh_ckpt} --gossip_checkpoint {gossip_ckpt} --test_dataset {dataset_name} --test_gossip --output_path {output_pretrained}'
    exit_code = os.system(command_pretrained)
    if exit_code == 0:
        # extract runtime and sampling time from pretrained
        try:
            patterns_pretrained = {
                'start': r'\[(.*?)\]: DeSCo Start\.',
                'end': r'\[(.*?)\]: DeSCo End\.',
                'neigh_application_start': r'\[(.*?)\]: Start Applying Neighborhood Count to Gossip\.',
                'neigh_application_end': r'\[(.*?)\]: Neighborhood Count Applied to Gossip\.',
                'output_start': r'\[(.*?)\]: Start Outputting and Analyzing Prediction Results\.',
                'output_end': r'\[(.*?)\]: Prediction Results Outputted and Analyzed\.'
            }
            log_pretrained = open(os.path.join(output_pretrained, 'log.txt'), 'r').read()
            timestamps = {}
            for key, pattern in patterns_pretrained.items():
                match = re.search(pattern, log_pretrained)
                if match:
                    timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")

            total_time = (timestamps['end'] - timestamps['start']).total_seconds()
            apply_time = (timestamps['neigh_application_end'] - timestamps['neigh_application_start']).total_seconds()
            output_time = (timestamps['output_end'] - timestamps['output_start']).total_seconds()
            runtime = total_time - apply_time - output_time

            # save the runtime and sampling time into time.txt
            with open(os.path.join(output_pretrained, 'time.txt'), 'w') as file:
                file.write(f'{runtime} {0}\n')
                file.write('Format: runtime sampling_time')

            # convert the prediction_graph.csv to prediction.csv by exchanging the columns
            df = pd.read_csv(os.path.join(output_pretrained, 'prediction_graph.csv'), index_col=0)
            df_new = df.iloc[:, column_exchange]
            df_new.columns = df.columns
            df_new.to_csv( os.path.join(output_pretrained, 'prediction.csv'))
        except Exception as e:
            print("Error in extracting runtime and sampling time from pretrained: ", e)
            with open(os.path.join(output_pretrained, 'error_extract_time.txt'), 'w') as file:
                file.write(str(e))
    else:
        print("Pretrain command failed, error code: ", exit_code)
        with open(os.path.join(output_pretrained, 'error_run_command.txt'), 'w') as file:
            file.write(str(exit_code))

def get_nei_gossip_ckpt(idx):
    if idx == 0:
        return 'ckpt/neighborhood_counting.ckpt', 'ckpt/gossip_propagation.ckpt'
    elif 1 <= idx <= 38:
        return f'ckpt/DeSCo/Syn_1827/neigh/lightning_logs/version_1/checkpoints/neighborhood_epochepoch={idx-1}.ckpt', 'ckpt/gossip_propagation.ckpt'
    elif 38 <= idx <= 42:
        return f'ckpt/DeSCo/Syn_1827/neigh/lightning_logs/version_1/checkpoints/neighborhood_epochepoch=37.ckpt', f'ckpt/DeSCo/Syn_1827/gossip/lightning_logs/version_1/checkpoints/gossip_epochepoch={idx-39}.ckpt'
    else:
        raise ValueError(f'Invalid index {idx}')

def main(input_path, output_path):
    dataset_name = f'Set_1'
    output_folder = os.path.join(output_path, dataset_name, 'DeSCo')
    # change the directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 2. use the pretrained model to predict the output
    for idx in range(38,43):
        run_pretrained(output_folder, dataset_name, idx)

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)