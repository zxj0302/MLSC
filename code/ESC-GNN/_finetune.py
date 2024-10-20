import os
import os.path as osp
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, GINEConv
import torch_geometric.transforms as T
from GraphCountDataset import dataset_random_graph
import os.path as osp
import os, sys
from shutil import copy, rmtree
import pdb
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, GELU, BatchNorm1d as BN, Dropout
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool, GATConv
from utils_edge_efficient import create_subgraphs
from sklearn.metrics import mean_absolute_error as MAE
from modules.ppgn_modules import *
from torch_geometric.utils import degree, dropout_adj, to_dense_batch, to_dense_adj
import json
from dataloader import DataLoader

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = "ESC-GNN"
NUM_EPOCHS = 1000
NUM_TARGETS = 29
TARGETS = list(range(1,29))

class NestedGIN_eff(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, use_z=False, use_rd=False, use_cycle=False, graph_pred=True,
                 use_id=None, dropout=0.2, multi_layer=False, edge_nest=False):
        super(NestedGIN_eff, self).__init__()
        self.use_rd = use_rd
        self.use_z = True
        self.graph_pred = graph_pred  # delete the final graph-level pooling
        self.use_cycle = use_cycle  # to mark whther predicting the cycle or not
        self.use_id = use_id
        self.dropout = dropout  # dropout 0.1 for multilayer, 0.2 for no multi
        self.multi_layer = multi_layer  # to use multi layer supervision or not
        self.edge_nest = edge_nest  # denote whether using the edge-level nested information
        z_in = 1800# if self.use_rd else 1700
        emb_dim = hidden
        self.z_initial = torch.nn.Embedding(z_in, emb_dim)
        self.z_embedding = Sequential(Dropout(dropout),
                                      torch.nn.BatchNorm1d(emb_dim),
                                      ReLU(),
                                      Linear(emb_dim, emb_dim),
                                      Dropout(dropout),
                                      torch.nn.BatchNorm1d(emb_dim),
                                      ReLU()
                                      )
        input_dim = 10#1800#dataset.num_features
        #if self.use_z or self.use_rd:
        #    input_dim += 8
        self.x_embedding = Sequential(Linear(input_dim, hidden),
                           Dropout(dropout),
                           BN(hidden),
                           ReLU(),
                           Linear(hidden, hidden),
                           Dropout(dropout),
                           BN(hidden),
                           ReLU()
        )
        if use_id is None:
            # self.conv1 = GCNConv(input_dim, hidden)

            self.conv1 = GINEConv(
                Sequential(
                    Linear(input_dim, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                ),
                train_eps=True,
                edge_dim = hidden)

            #self.conv1 = GATConv(input_dim, hidden, edge_dim = hidden, add_self_loops = False)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if use_id is None:

                self.convs.append(GINEConv(
                    Sequential(
                        Linear(hidden, hidden),
                        Dropout(dropout),
                        BN(hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        Dropout(dropout),
                        BN(hidden),
                        ReLU(),
                    ),
                    train_eps=True,
                    edge_dim = hidden))

                #self.convs.append(GATConv(hidden, hidden, edge_dim = hidden, add_self_loops = False))

        self.lin1 = torch.nn.Linear(num_layers * hidden + hidden, hidden)
        #self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        #self.lin1 = torch.nn.Linear(hidden, hidden)
        self.bn_lin1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        if not use_cycle:
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.lin2 = Linear(hidden, 1)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for layer in self.z_embedding.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.bn_lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        data.to(self.lin1.weight.device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        #edge_pos[:, :200] = 0
        #edge_pos[:, 200:500] = 0
        #edge_pos[:, -1300:] = 0
        
        if hasattr(data, 'edge_pos'):
            # original, slow version
            edge_pos = data.edge_pos.float()
            z_emb = torch.mm(edge_pos, self.z_initial.weight)
        else:
            # new, fast version
            
            # for ablation study
            #mask_index = (data.pos_index >= 500)
            #mask_index = torch.logical_and((data.pos_index >= 200), (data.pos_index < 500))
            #mask_index = (data.pos_index < 500)
            #z_emb = global_add_pool(torch.mul(self.z_initial.weight[data.pos_index[~mask_index]], data.pos_enc[~mask_index].view(-1, 1)), data.pos_batch[~mask_index])
            
            z_emb = global_add_pool(torch.mul(self.z_initial.weight[data.pos_index], data.pos_enc.view(-1, 1)), data.pos_batch)
        z_emb = self.z_embedding(z_emb)
        
        #z_emb = self.z_embedding(edge_pos)

        if self.use_id is None:
            x = self.conv1(x, edge_index, z_emb)
        else:
            x = self.conv1(x, edge_index, data.node_id)

        #xs = [x]
        xs = [self.x_embedding(data.x), x]
        for conv in self.convs:
            if self.use_id is None:
                x = conv(x, edge_index, z_emb)
            else:
                if self.edge_nest:
                    x = conv(x, edge_index, data.node_id) + conv(x, edge_index, data.node_id + 1)
                else:
                    x = conv(x, edge_index, data.node_id)
            xs += [x]

        if self.graph_pred:
            # x = global_add_pool(x, data.batch)
            x = global_mean_pool(torch.cat(xs, dim = 1), batch)
        else:
            x = torch.cat(xs, dim = 1)
            #x = x
        x = self.lin1(x)
        if x.size()[0] > 1:
            x = self.bn_lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        if not self.use_cycle:
            return F.log_softmax(x, dim=-1)
        else:
            return x#, []

def load_model_from_checkpoint(checkpoint_dir, cpt=2000):
    try:
        folder_names = os.listdir(checkpoint_dir)
        folder_names.sort()
        if len(folder_names) == 0:
            raise FileNotFoundError(f"No checkpoint folder found in {checkpoint_dir}")
        checkpoint_dir = os.path.join(checkpoint_dir, folder_names[-1])

        args_path = os.path.join(checkpoint_dir, "args.json")
        checkpoint_path = os.path.join(checkpoint_dir, f"cpt_{cpt}.pth")

        if not os.path.exists(args_path):
            raise FileNotFoundError(f"args.json not found in the checkpoint directory: {args_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        with open(args_path, "r") as f:
            saved_args = json.load(f)

        args = argparse.Namespace(**saved_args)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model = NestedGIN_eff(
            dataset=args.dataset,
            num_layers=args.layers,
            hidden=256, use_rd=True, graph_pred=False, dropout=0, edge_nest=True, use_cycle=True
        )
        model.load_state_dict(checkpoint)
        model.eval()

        return model, args

    except Exception as e:
        logger.error(f"Error loading model from checkpoint: {str(e)}")
        raise


def load_dataset(args, dataset_name, split="test"):
    """
    Load and preprocess a dataset.
    """
    def MyTransform(data):
        data.y = data.y[:, int(args.target)]
        return data

    dataname = dataset_name
    processed_name = dataname
    if args.h is not None:
        processed_name = processed_name + "_h" + str(args.h)
    
    path = 'data/Count'

    pre_transform = None
    if args.h is not None:
        if type(args.h) == int:
            path += '/ngnn_h' + str(args.h)
        elif type(args.h) == list:
            path += '/ngnn_h' + ''.join(str(h) for h in args.h)
        path += '_' + args.node_label
        if args.max_nodes_per_hop is not None:
            path += '_mnph{}'.format(args.max_nodes_per_hop)
        def pre_transform(g):
            return create_subgraphs(g, args.h,
                                    max_nodes_per_hop=args.max_nodes_per_hop,
                                    node_label=args.node_label,
                                    use_rd=True, self_loop = True)
    pre_filter = None
    if args.model == "NestedGIN_eff":
        my_pre_transform = pre_transform
    else:
        my_pre_transform = None

    dataset = dataset_random_graph(
        dataname=dataset_name,
        processed_name=processed_name,
        transform=MyTransform,
        pre_transform=my_pre_transform,
        split=split,
    )

    return dataset

def finetune_model(model, args, train_dataset, val_dataset, device, num_epochs=300, dataset_name="Set_1"):
    """
    Finetune the model on the given dataset with validation.
    """
    try:
        model.train()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.patience,
            min_lr=0.00001,
        )
        print(f"Finetuning model on dataset {dataset_name} for target {args.target}")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        best_val_error = None
        for epoch in tqdm(range(1, num_epochs+1), desc="Finetuning"):
            # Training
            model.train()
            total_loss = 0
            for data in train_loader:
                if type(data) == dict:
                    data = {key: data_.to(device) for key, data_ in data.items()}
                else:
                    data = data.to(device)
                optimizer.zero_grad()
    
                y = data.y.view([data.y.size(0), 1])
                Loss = torch.nn.L1Loss()
                loss = Loss(model(data), y)
                loss.backward()

                total_loss += loss.item() * y.size(0)
                optimizer.step()

            avg_loss = total_loss / train_dataset.data.y.size(0)

            # Validation
            val_error = test(val_loader, model, args, device)
            scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                best_val_error = val_error

            # Logging
            lr = scheduler.optimizer.param_groups[0]["lr"]
            if epoch % 25 == 0:
                logger.info(
                    f"Epoch: {epoch:03d}, LR: {lr:.7f}, Loss: {avg_loss:.7f}, Validation MAE: {val_error:.7f}"
                )
            if epoch <= NUM_EPOCHS:
                save_model(model, args, f"/workspace/output/final_fine/{dataset_name}/{MODEL}/{args.target}", args.target, epoch)

        return model

    except Exception as e:
        logger.error(f"Error during finetuning: {str(e)}")
        raise

def test(loader, model, args, device):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y = data.y.view([data.y.size(0), 1])
            error += torch.sum(torch.abs(out - y)).item()
    return error / len(loader.dataset)

def save_model(model, args, save_dir, target, epoch=300):
    """
    Save the finetuned model.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"finetuned_{epoch}.pth")
        torch.save(model.state_dict(), save_path)

        args_path = os.path.join(save_dir, f"finetuned_args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info(f"Finetuned model and args saved for target {target}")

    except Exception as e:
        logger.error(f"Error saving finetuned model: {str(e)}")
        raise

def main():
    base_save_dir = "/workspace/output/final_fine/"
    os.makedirs(base_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    base_checkpoint_dir = f"/workspace/output/data_esc/{MODEL}/Checkpoints"
    for finetune_dataset_name in ["Set_1"]:
        for target in TARGETS:
            try:
                checkpoint_dir = os.path.join(base_checkpoint_dir, str(target))
                
                # Load model and args
                model, args = load_model_from_checkpoint(checkpoint_dir)
                args.target = target  # Update target in args

                # Load finetuning datasets
                train_dataset = load_dataset(args, finetune_dataset_name, split="train")
                val_dataset = load_dataset(args, finetune_dataset_name, split="val")

                # Finetune model
                finetuned_model = finetune_model(
                    model,
                    args,
                    train_dataset,
                    val_dataset,
                    device,
                    num_epochs=NUM_EPOCHS,
                    dataset_name=finetune_dataset_name,
                )

                # Save finetuned model
                save_dir = os.path.join(base_save_dir, finetune_dataset_name, MODEL, str(target))
                save_model(finetuned_model, args, save_dir, target, epoch=NUM_EPOCHS)

            except Exception as e:
                logger.error(f"Error processing target {target}: {str(e)}")
                continue

def dataset_load_time():
    TARGET_DIAM = [2, 1,  3, 2, 2, 2, 2, 1,   4, 3, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    import time
    args_path = "/workspace/output/data_esc/ESC-GNN/Checkpoints/0/20241010215852/args.json"
    with open(args_path, "r") as f:
        saved_args = json.load(f)

    # Create args object and populate it
    args = type("Args", (), {})()
    for key, value in saved_args.items():
        setattr(args, key, value)

    res = json.load(open("/workspace/output/load_time.json", "r"))
    for dataset in [f"Set_{i}" for i in range(1, 2)]:
        res[dataset] = {} if dataset not in res else res[dataset]
        for alg in ["ESC-GNN"]:
            res[dataset][alg] = {} if alg not in res[dataset] else res[dataset][alg]
            for target in [8]:
                res[dataset][alg][target] = {} if target not in res[dataset][alg] else res[dataset][alg][target]
                # update args
                args.dataset = dataset
                args.target = target
                args.h = TARGET_DIAM[target]
                
                for split in ["train", "val", "test"]:
                    if split in res[dataset][alg][target]:
                        continue
                    print(f"Loading dataset {dataset} split {split}...")
                    start = time.time()
                    data = load_dataset(args, dataset, split=split)
                    print(f"Loaded dataset {dataset} split {split} in {time.time() - start:.2f} seconds.")
                    res[dataset][alg][target][split] = time.time() - start
                    # save res
                    with open(f"/workspace/output/load_time.json", "w") as f:
                        json.dump(res, f, indent=2)


if __name__ == "__main__":
    dataset_load_time()