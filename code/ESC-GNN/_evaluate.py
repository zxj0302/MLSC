import torch

from dataloader import DataLoader

# Import model definitions
# from count_models import GNN, PPGN, NGNN, GNNAK, IDGNN, I2GNN
# Import dataset processing (you might need to adjust this import based on your project structure)
from utils import create_subgraphs
import os
import numpy as np
import torch
from GraphCountDataset import dataset_random_graph
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, GELU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool, GATConv
from utils_edge_efficient import create_subgraphs
from sklearn.metrics import mean_absolute_error as MAE
from modules.ppgn_modules import *
import json



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

def load_model_from_checkpoint(checkpoint_dir, cpt=100):
    """
    Load a model from a checkpoint directory containing args.json and cpt_2000.pth.
    """
    # find the folder name with the highest timestamp
    folder_names = os.listdir(checkpoint_dir)
    # sort the folder names by timestamp
    folder_names.sort()
    if len(folder_names) == 0:
        raise FileNotFoundError(f"No checkpoint folder found in {checkpoint_dir}")
    checkpoint_dir = os.path.join(checkpoint_dir, folder_names[-1])
    print(checkpoint_dir)


    args_path = os.path.join(checkpoint_dir, "args.json")
    checkpoint_path = os.path.join(checkpoint_dir, f"cpt_{cpt}.pth")

    if not os.path.exists(args_path):
        raise FileNotFoundError(
            f"args.json not found in the checkpoint directory: {args_path}"
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with open(args_path, "r") as f:
        saved_args = json.load(f)

    args = type("Args", (), saved_args)()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_class = NestedGIN_eff

    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model_class(
        "data_esc",
        num_layers=3,
        hidden=256, use_rd = True, graph_pred = False, dropout = 0, edge_nest = True, use_cycle = True
    )
    model.load_state_dict(checkpoint)
    model.eval()

    return model, args


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


def test(loader, model, args, device, mean, std, output=False):
    model.eval()
    error = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in loader:
            if isinstance(data, dict):
                data = {key: data_.to(device) for key, data_ in data.items()}
            else:
                data = data.to(device)

            y = data.y
            y_hat = model(data)[:, 0]
            print(y.shape, y_hat.shape)
            # Denormalize
            # y_hat = y_hat * std[args.target] + mean[args.target]

            error += torch.sum(torch.abs(y_hat - y)).item()
            if output:
                y_pred.extend(y_hat.cpu().numpy().tolist())
                y_true.extend(y.cpu().numpy().tolist())

    mae = error / len(loader.dataset)

    if output:
        return mae, y_pred, y_true
    return mae


def run_test_on_dataset(model, args, dataset_name):
    """
    Run test on a given dataset using the loaded model.
    """
    dataset = load_dataset(args, dataset_name)
    test_loader = DataLoader(dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Running test on {len(test_loader.dataset)} samples...")

    # dataset_name = "data_esc"
    train_dataset = load_dataset(args, dataset_name, split="train")
    std = train_dataset.data.y.std(dim=0)
    mean = train_dataset.data.y.mean(dim=0)

    mae, y_pred, y_true = test(test_loader, model, args, device, mean, std, output=True)
    # calculate MAE
    # mae = np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
    gt_mean = np.mean(y_true)
    gt_std = np.std(y_true)
    pred_mean = np.mean(y_pred) 
    pred_std = np.std(y_pred)


    return {"mae": mae, "gt_mean": gt_mean, "gt_std": gt_std, "pred_mean": pred_mean, "pred_std": pred_std, "predictions": y_pred, "ground_truth": y_true}

def zeroshot():
    MODEL = "ESC-GNN"
    TARGET = list(range(0,29))
    SET = 1
    for t in TARGET:
        # /home/zxj/Dev/MLSC/output/final_fine/Set_1/ESC-GNN/1/finetuned_300.pth
        # /home/zxj/Dev/MLSC/output/data_esc/ESC-GNN/Checkpoints/0
        checkpoint_dir = f"/workspace/output/data_esc/ESC-GNN/Checkpoints/{t}"
        print(f"Loading model from checkpoint: {checkpoint_dir}")
        model, args = load_model_from_checkpoint(checkpoint_dir, cpt=2000)
        print(f"Model loaded: {type(model).__name__}")

        test_dataset_name = "Set_1"  # Replace with your actual test dataset name
        print(f"Running test on dataset: {test_dataset_name}")
        test_results = run_test_on_dataset(model, args, test_dataset_name)

        print(f"Test Results:")
        print(f"  MAE: {test_results['mae']}")
        print(f"  Number of samples: {len(test_results['predictions'])}")

        # Optionally, save the results to a file
        output_dir = f"output/zero/{test_dataset_name}/{MODEL}/{t}/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{2000}_cpt_test.json")
        with open(output_file, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {output_file}")


def main():
    MODEL = "ESC-GNN"
    TARGET = [13]
    SET = 1
    for cpt in range(100, 2001, 100):
        # /home/zxj/Dev/MLSC/output/final_fine/Set_1/ESC-GNN/1/finetuned_300.pth
        checkpoint_dir = f"/workspace/output/retrain/Set_1/ESC-GNN/Checkpoints/11"
        print(f"Loading model from checkpoint: {checkpoint_dir}")
        model, args = load_model_from_checkpoint(checkpoint_dir, cpt=cpt)
        print(f"Model loaded: {type(model).__name__}")

        test_dataset_name = "Set_1"  # Replace with your actual test dataset name
        print(f"Running test on dataset: {test_dataset_name}")
        test_results = run_test_on_dataset(model, args, test_dataset_name)

        print(f"Test Results:")
        print(f"  MAE: {test_results['mae']}")
        print(f"  Number of samples: {len(test_results['predictions'])}")

        # Optionally, save the results to a file
        output_dir = f"output/retrain/{test_dataset_name}/{MODEL}/11/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{cpt}_cpt_test.json")
        with open(output_file, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {output_file}")



if __name__ == "__main__":
    main()