# Note: this script should be in the same directory as the main.py
# TODO: print status messages to the console
import re
import os
import sys
import torch
import shutil
from torch_geometric.data import InMemoryDataset, Data

# get the sampled data path and output path from the command line
input_path = sys.argv[1] # MLSC/input/
output_path = sys.argv[2] # MLSC/output/

# for all the folders named Set_i in the sampled_data_path folder, do the following:
for i in range(1, len([s for s in os.listdir(input_path) if re.match(r"Set_\d+$", s)]) + 1):
    dataset_name = f'Set_{i}'
    output_folder = os.path.join(output_path, dataset_name, 'DeSCo')
    # convert it to the correct format
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

    # use the model to predict the output
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. use the pretrained model to predict the output
    command_pretrained = f'python main.py --neigh_checkpoint ckpt/neighborhood_counting.ckpt --gossip_checkpoint ckpt/gossip_propagation.ckpt --test_dataset {dataset_name} --test_gossip --output_path {os.path.abspath(os.path.join(output_folder, "pretrained"))}'
    os.system(command_pretrained)

    # 2. fine-tune the model
    command_finetune = (f'python main.py --neigh_checkpoint ckpt/neighborhood_counting.ckpt --gossip_checkpoint '
                        f'ckpt/gossip_propagation.ckpt --train_dataset {dataset_name} --valid_dataset'
                        f' {dataset_name} --test_dataset {dataset_name} --train_neigh --train_gossip --test_gossip '
                        f'--output_path {os.path.abspath(os.path.join(output_folder, "finetuning"))} '
                        f'--neigh_epoch_num 300 --gossip_epoch_num 30 --gossip_batch_size 20')
    os.system(command_finetune)
    # read in the fine-tuned model checkpoint path
    neigh_pattern = r"Best Neighborhood Model Path: (ckpt/.*?\.ckpt)"
    gossip_pattern = r"Best Gossip Model Path: (ckpt/.*?\.ckpt)"
    log_finetuning = open(os.path.join(output_folder, 'finetuning', 'log.txt')).read()
    neigh_checkpoint_finetuned = re.search(neigh_pattern, log_finetuning).group(1)
    gossip_checkpoint_finetuned = re.search(gossip_pattern, log_finetuning).group(1)

    # 3. use the fine-tuned model to predict the output
    command_finetuned = f'python main.py --neigh_checkpoint {neigh_checkpoint_finetuned} --gossip_checkpoint {gossip_checkpoint_finetuned} --test_dataset {dataset_name} --test_gossip --output_path {os.path.abspath(os.path.join(output_folder, "finetuned"))}'
    os.system(command_finetuned)

    # 4. extract info and convert output result
    # 4.1 extract runtime and sampling time from pretrained
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
    log_content = open(os.path.join(output_folder, 'pretrained', 'log.txt'), 'r').read()
    timestamps = {}
    for key, pattern in patterns_pretrained.items():
        match = re.search(pattern, log_content)
        if match:
            timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")

    total_time = (timestamps['end'] - timestamps['start']).total_seconds()
    sampling_conversion_time = (timestamps['conversion_end'] - timestamps['sampling_start']).total_seconds()
    apply_time = (timestamps['neigh_application_end'] - timestamps['neigh_application_start']).total_seconds()
    output_time = (timestamps['output_end'] - timestamps['output_start']).total_seconds()

    runtime = total_time - sampling_conversion_time - apply_time - output_time
    sampling_time = (timestamps['sampling_end'] - timestamps['sampling_start']).total_seconds()

    # save the runtime and sampling time into time.txt
    with open(os.path.join(output_folder, 'pretrained', 'time.txt'), 'w') as file:
        file.write(f'{runtime} {sampling_time}\n')
        file.write('Format: runtime sampling_time')
    print(runtime)

    # 4.2 extract runtime from fine-tuned
    patterns_finetuned = {
        'start': r'\[(.*?)\]: DeSCo Start\.',
        'end': r'\[(.*?)\]: DeSCo End\.',
        'neigh_application_start': r'\[(.*?)\]: Start Applying Neighborhood Count to Gossip\.',
        'neigh_application_end': r'\[(.*?)\]: Neighborhood Count Applied to Gossip\.',
        'output_start': r'\[(.*?)\]: Start Outputting and Analyzing Prediction Results\.',
        'output_end': r'\[(.*?)\]: Prediction Results Outputted and Analyzed\.'
    }
    log_content = open(os.path.join(output_folder, 'finetuned', 'log.txt'), 'r').read()
    timestamps = {}
    for key, pattern in patterns_finetuned.items():
        match = re.search(pattern, log_content)
        if match:
            timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")

    total_time = (timestamps['end'] - timestamps['start']).total_seconds()
    apply_time = (timestamps['neigh_application_end'] - timestamps['neigh_application_start']).total_seconds()
    output_time = (timestamps['output_end'] - timestamps['output_start']).total_seconds()
    runtime = total_time - apply_time - output_time

    # 4.3 extract training time from fine-tuning
    patterns_finetuning = {
        'neigh_start': r'\[(.*?)\]: Start Training Neighborhood Model\.',
        'neigh_end': r'\[(.*?)\]: Neighborhood Model Trained\.',
        'gossip_start': r'\[(.*?)\]: Start Training Gossip Model\.',
        'gossip_end': r'\[(.*?)\]: Gossip Model Trained\.'
    }
    log_content = open(os.path.join(output_folder, 'finetuning', 'log.txt'), 'r').read()
    timestamps = {}
    for key, pattern in patterns_finetuning.items():
        match = re.search(pattern, log_content)
        if match:
            timestamps[key] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
    neigh_train_time = (timestamps['neigh_end'] - timestamps['neigh_start']).total_seconds()
    gossip_train_time = (timestamps['gossip_end'] - timestamps['gossip_start']).total_seconds()
    training_time = neigh_train_time + gossip_train_time

    # 4.4 save the runtime, training time, neighborhood training time, and gossip training time into time.txt
    with open(os.path.join(output_folder, 'finetuned', 'time.txt'), 'w') as file:
        file.write(f'{runtime} {training_time} {neigh_train_time} {gossip_train_time}\n')
        file.write('Format: runtime training_time neigh_train_time gossip_train_time')

    # 4.5 convert the prediction_graph.csv to prediction.csv
    column_exchange = [0, 1, 3, 2, 5, 4, 6, 7, 10, 9, 8, 13, 11, 12, 15, 14, 16, 18, 17, 20, 19, 22, 21, 23, 24, 25, 26, 27, 28]
    for process in ['pretrained', 'finetuned']:
        prediction_graph_path = os.path.join(output_folder, process, 'prediction_graph.csv')
        df = pd.read_csv(prediction_graph_path, index_col=0)
        df_new = df.iloc[:, column_exchange]
        df_new.columns = df.columns
        df_new.to_csv( os.path.join(output_folder, process, 'prediction.csv'))