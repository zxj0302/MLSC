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
                        f'--neigh_epoch_num 10 --gossip_epoch_num 5')
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