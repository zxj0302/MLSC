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

column_exchange = [0, 1, 3, 2, 5, 4, 6, 7, 10, 9, 8, 13, 11, 12, 15, 14, 16, 18, 17, 20, 19, 22, 21, 23, 24, 25, 26, 27, 28]

def dataset_conversion(input_path, dataset_name):
    sampled_data = torch.load(os.path.join(input_path, dataset_name, 'dataset.pt'))
    for split in ['train', 'val', 'test']:
        sampled_data_clean = [Data(edge_index=s.edge_index, x=torch.ones(s.num_nodes, 1)) for s in sampled_data[split]]
        ground_truth = torch.cat([s.gt_induced_le5_desco for s in sampled_data[split]], dim=0)
        # check whether data/data_name exists, if not create it
        split_path = os.path.join('data', dataset_name, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
        os.makedirs(os.path.join(split_path, 'processed'))
        os.makedirs(os.path.join(split_path, 'raw'))
        os.makedirs(os.path.join(split_path, 'CanonicalCountTruth'))
        torch.save(InMemoryDataset.collate(sampled_data_clean), os.path.join(split_path, 'processed', 'rwd.pt'))
        torch.save(ground_truth, os.path.join(split_path, 'CanonicalCountTruth', 'query_num_29_query_len_sum_135.pt'))

def run_pretrained(output_folder, dataset_name):
    output_pretrained = os.path.abspath(os.path.join(output_folder, "pretrained"))
    if os.path.exists(output_pretrained):
        os.system(f'rm -rf {output_pretrained}/*')
    command_pretrained = f'python main.py --neigh_checkpoint ckpt/neighborhood_counting.ckpt --gossip_checkpoint ckpt/gossip_propagation.ckpt --test_dataset {dataset_name} --test_gossip --output_path {output_pretrained}'
    exit_code = os.system(command_pretrained)
    if exit_code == 0:
        # extract runtime and sampling time from pretrained
        try:
            patterns_pretrained = {
                'start': r'\[(.*?)\]: DeSCo Start\.',
                'end': r'\[(.*?)\]: DeSCo End\.',
                'sampling_start': r'\[(.*?)\]: Start Sampling Neighbors\.',
                'sampling_end': r'\[(.*?)\]: Neighbors Sampling Done\.',
                'conversion_start': r'\[(.*?)\]: Start Converting to PYG Graph\.',
                'conversion_end': r'\[(.*?)\]: Conversion to PYG Graph Done\.',
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
            sampling_conversion_time = (timestamps['conversion_end'] - timestamps['sampling_start']).total_seconds()
            apply_time = (timestamps['neigh_application_end'] - timestamps['neigh_application_start']).total_seconds()
            output_time = (timestamps['output_end'] - timestamps['output_start']).total_seconds()
            runtime = total_time - sampling_conversion_time - apply_time - output_time
            sampling_time = (timestamps['sampling_end'] - timestamps['sampling_start']).total_seconds()

            # save the runtime and sampling time into time.txt
            with open(os.path.join(output_pretrained, 'time.txt'), 'w') as file:
                file.write(f'{runtime} {sampling_time}\n')
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

def run_finetuned(output_folder, dataset_name, neigh_epoch_num, gossip_epoch_num, gossip_batch_size):
    output_finetuning = os.path.abspath(os.path.join(output_folder, "finetuning"))
    if os.path.exists(output_finetuning):
        os.system(f'rm -rf {output_finetuning}/*')
    command_finetuning = f'python main.py --neigh_checkpoint ckpt/neighborhood_counting.ckpt --gossip_checkpoint ckpt/gossip_propagation.ckpt --train_dataset {dataset_name} --valid_dataset {dataset_name} --test_dataset {dataset_name} --train_neigh --train_gossip --test_gossip --output_path {output_finetuning} --neigh_epoch_num {neigh_epoch_num} --gossip_epoch_num {gossip_epoch_num} --gossip_batch_size {gossip_batch_size}'
    exit_code = os.system(command_finetuning)
    if exit_code == 0:
        # extract training time from fine-tuning
        try:
            patterns_finetuning = {
                'neigh_start': r'\[(.*?)\]: Start Training Neighborhood Model\.',
                'neigh_end': r'\[(.*?)\]: Neighborhood Model Trained\.',
                'gossip_start': r'\[(.*?)\]: Start Training Gossip Model\.',
                'gossip_end': r'\[(.*?)\]: Gossip Model Trained\.'
            }
            log_finetuning = open(os.path.join(output_finetuning, 'log.txt'), 'r').read()
            timestamps = {}
            for key, pattern in patterns_finetuning.items():
                match = re.search(pattern, log_finetuning)
                if match:
                    timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
            neigh_train_time = (timestamps['neigh_end'] - timestamps['neigh_start']).total_seconds()
            gossip_train_time = (timestamps['gossip_end'] - timestamps['gossip_start']).total_seconds()
            training_time = neigh_train_time + gossip_train_time
        except Exception as e:
            print("Error in extracting training time from fine-tuning: ", e)
            with open(os.path.join(output_finetuning, 'error_extract_time.txt'), 'w') as file:
                file.write(str(e))

        # read in the fine-tuned model checkpoint path
        try:
            neigh_pattern = r"Best Neighborhood Model Path: (ckpt/.*?\.ckpt)"
            gossip_pattern = r"Best Gossip Model Path: (ckpt/.*?\.ckpt)"
            neigh_checkpoint_finetuned = re.search(neigh_pattern, log_finetuning).group(1)
            gossip_checkpoint_finetuned = re.search(gossip_pattern, log_finetuning).group(1)

            # use the fine-tuned model to predict the output
            output_finetuned = os.path.abspath(os.path.join(output_folder, "finetuned"))
            if os.path.exists(output_finetuned):
                os.system(f'rm -rf {output_finetuned}/*')
            command_finetuned = f'python main.py --neigh_checkpoint {neigh_checkpoint_finetuned} --gossip_checkpoint {gossip_checkpoint_finetuned} --test_dataset {dataset_name} --test_gossip --output_path {output_finetuned}'
            exit_code = os.system(command_finetuned)
            if exit_code == 0:
                # extract runtime from fine-tuned
                try:
                    patterns_finetuned = {
                        'start': r'\[(.*?)\]: DeSCo Start\.',
                        'end': r'\[(.*?)\]: DeSCo End\.',
                        'neigh_application_start': r'\[(.*?)\]: Start Applying Neighborhood Count to Gossip\.',
                        'neigh_application_end': r'\[(.*?)\]: Neighborhood Count Applied to Gossip\.',
                        'output_start': r'\[(.*?)\]: Start Outputting and Analyzing Prediction Results\.',
                        'output_end': r'\[(.*?)\]: Prediction Results Outputted and Analyzed\.'
                    }
                    log_finetuned = open(os.path.join(output_finetuned, 'log.txt'), 'r').read()
                    timestamps = {}
                    for key, pattern in patterns_finetuned.items():
                        match = re.search(pattern, log_finetuned)
                        if match:
                            timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")

                    total_time = (timestamps['end'] - timestamps['start']).total_seconds()
                    apply_time = (timestamps['neigh_application_end'] - timestamps['neigh_application_start']).total_seconds()
                    output_time = (timestamps['output_end'] - timestamps['output_start']).total_seconds()
                    runtime = total_time - apply_time - output_time

                    # save the runtime, training time, neighborhood training time, and gossip training time into time.txt
                    with open(os.path.join(output_finetuned, 'time.txt'), 'w') as file:
                        file.write(f'{runtime} {training_time} {neigh_train_time} {gossip_train_time}\n')
                        file.write('Format: runtime training_time neigh_train_time gossip_train_time')

                    # convert the prediction_graph.csv to prediction.csv by exchanging the columns
                    df = pd.read_csv(os.path.join(output_finetuned, 'prediction_graph.csv'), index_col=0)
                    df_new = df.iloc[:, column_exchange]
                    df_new.columns = df.columns
                    df_new.to_csv( os.path.join(output_finetuned, 'prediction.csv'))
                except Exception as e:
                    print("Error in extracting runtime from fine-tuned: ", e)
                    with open(os.path.join(output_finetuned, 'error_extract_time.txt'), 'w') as file:
                        file.write(str(e))
            else:
                print("Finetuned command failed, error code: ", exit_code)
                with open(os.path.join(output_finetuned, 'error_run_command.txt'), 'w') as file:
                    file.write(str(exit_code))
        except Exception as e:
            print("Error in extracting fine-tuned model checkpoint path: ", e)
            with open(os.path.join(output_finetuning, 'error_extract_checkpoint.txt'), 'w') as file:
                file.write(str(e))
    else:
        print("Finetuning command failed, error code: ", exit_code)
        with open(os.path.join(output_finetuning, 'error_run_command.txt'), 'w') as file:
            file.write(str(exit_code))

def run_retrained(output_folder, dataset_name, neigh_epoch_num, gossip_epoch_num, gossip_batch_size):
    output_retraining = os.path.abspath(os.path.join(output_folder, "retraining"))
    if os.path.exists(output_retraining):
        os.system(f'rm -rf {output_retraining}/*')
    command_retraining = (f'python main.py --train_dataset {dataset_name} --valid_dataset {dataset_name} --test_dataset {dataset_name} --train_neigh --train_gossip --test_gossip --output_path {output_retraining} --neigh_epoch_num {neigh_epoch_num} --gossip_epoch_num {gossip_epoch_num} --gossip_batch_size {gossip_batch_size}')
    exit_code = os.system(command_retraining)
    if exit_code == 0:
        # extract training time from re-training
        try:
            patterns_retraining = {
                'neigh_start': r'\[(.*?)\]: Start Training Neighborhood Model\.',
                'neigh_end': r'\[(.*?)\]: Neighborhood Model Trained\.',
                'gossip_start': r'\[(.*?)\]: Start Training Gossip Model\.',
                'gossip_end': r'\[(.*?)\]: Gossip Model Trained\.'
            }
            log_retraining = open(os.path.join(output_retraining, 'log.txt'), 'r').read()
            timestamps = {}
            for key, pattern in patterns_retraining.items():
                match = re.search(pattern, log_retraining)
                if match:
                    timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
            neigh_train_time = (timestamps['neigh_end'] - timestamps['neigh_start']).total_seconds()
            gossip_train_time = (timestamps['gossip_end'] - timestamps['gossip_start']).total_seconds()
            training_time = neigh_train_time + gossip_train_time
        except Exception as e:
            print("Error in extracting training time from re-training: ", e)
            with open(os.path.join(output_retraining, 'error_extract_time.txt'), 'w') as file:
                file.write(str(e))

        # read in the re-trained model checkpoint path
        try:
            neigh_pattern = r"Best Neighborhood Model Path: (ckpt/.*?\.ckpt)"
            gossip_pattern = r"Best Gossip Model Path: (ckpt/.*?\.ckpt)"
            neigh_checkpoint_retrained = re.search(neigh_pattern, log_retraining).group(1)
            gossip_checkpoint_retrained = re.search(gossip_pattern, log_retraining).group(1)

            # use the re-trained model to predict the output
            output_retrained = os.path.abspath(os.path.join(output_folder, "retrained"))
            if os.path.exists(output_retrained):
                os.system(f'rm -rf {output_retrained}/*')
            command_retrained = f'python main.py --neigh_checkpoint {neigh_checkpoint_retrained} --gossip_checkpoint {gossip_checkpoint_retrained} --test_dataset {dataset_name} --test_gossip --output_path {output_retrained}'
            exit_code = os.system(command_retrained)
            if exit_code == 0:
                # extract runtime from re-trained
                try:
                    patterns_retrained = {
                        'start': r'\[(.*?)\]: DeSCo Start\.',
                        'end': r'\[(.*?)\]: DeSCo End\.',
                        'neigh_application_start': r'\[(.*?)\]: Start Applying Neighborhood Count to Gossip\.',
                        'neigh_application_end': r'\[(.*?)\]: Neighborhood Count Applied to Gossip\.',
                        'output_start': r'\[(.*?)\]: Start Outputting and Analyzing Prediction Results\.',
                        'output_end': r'\[(.*?)\]: Prediction Results Outputted and Analyzed\.'
                    }
                    log_retrained = open(os.path.join(output_retrained, 'log.txt'), 'r').read()
                    timestamps = {}
                    for key, pattern in patterns_retrained.items():
                        match = re.search(pattern, log_retrained)
                        if match:
                            timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")

                    total_time = (timestamps['end'] - timestamps['start']).total_seconds()
                    apply_time = (timestamps['neigh_application_end'] - timestamps['neigh_application_start']).total_seconds()
                    output_time = (timestamps['output_end'] - timestamps['output_start']).total_seconds()
                    runtime = total_time - apply_time - output_time

                    # save the runtime, training time, neighborhood training time, and gossip training time into time.txt
                    with open(os.path.join(output_retrained, 'time.txt'), 'w') as file:
                        file.write(f'{runtime} {training_time} {neigh_train_time} {gossip_train_time}\n')
                        file.write('Format: runtime training_time neigh_train_time gossip_train_time')

                    # convert the prediction_graph.csv to prediction.csv by exchanging the columns
                    df = pd.read_csv(os.path.join(output_retrained, 'prediction_graph.csv'), index_col=0)
                    df_new = df.iloc[:, column_exchange]
                    df_new.columns = df.columns
                    df_new.to_csv( os.path.join(output_retrained, 'prediction.csv'))
                except Exception as e:
                    print("Error in extracting runtime from re-trained: ", e)
                    with open(os.path.join(output_retrained, 'error_extract_time.txt'), 'w') as file:
                        file.write(str(e))
            else:
                print("Retrained command failed, error code: ", exit_code)
                with open(os.path.join(output_retrained, 'error_run_command.txt'), 'w') as file:
                    file.write(str(exit_code))
        except Exception as e:
            print("Error in extracting re-trained model checkpoint path: ", e)
            with open(os.path.join(output_retraining, 'error_extract_checkpoint.txt'), 'w') as file:
                file.write(str(e))
    else:
        print("Retraining command failed, error code: ", exit_code)
        with open(os.path.join(output_retraining, 'error_run_command.txt'), 'w') as file:
            file.write(str(exit_code))

def main(input_path, output_path):
    # for all the folders named Set_i in the sampled_data_path folder, do the following:
    for i in range(5, len([s for s in os.listdir(input_path) if re.match(r"Set_\d+$", s)]) + 1):
        dataset_name = f'Set_{i}'
        output_folder = os.path.join(output_path, dataset_name, 'DeSCo')
        # change the directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # 1. convert dataset to the needed format
        dataset_conversion(input_path, dataset_name)
        # 2. use the pretrained model to predict the output
        run_pretrained(output_folder, dataset_name)
        # 3. fine-tune the model
        neigh_epoch_num = 300
        gossip_epoch_num = 30
        gossip_batch_size = 20
        run_finetuned(output_folder, dataset_name, neigh_epoch_num, gossip_epoch_num, gossip_batch_size)
        # 4. re-train the model
        run_retrained(output_folder, dataset_name, neigh_epoch_num, gossip_epoch_num, gossip_batch_size)

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)