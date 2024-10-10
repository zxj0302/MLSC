import os
import json
import torch
import numpy as np

from dataloader import DataLoader
import torch.nn.functional as F

# Import model definitions
from count_models import GNN, PPGN, NGNN, GNNAK, IDGNN, I2GNN

# Import dataset processing (you might need to adjust this import based on your project structure)
import data_processing as dp
from utils import create_subgraphs, create_subgraphs2


def load_model_from_checkpoint(checkpoint_dir):
    """
    Load a model from a checkpoint directory containing args.json and cpt_2000.pth.
    """
    args_path = os.path.join(checkpoint_dir, "finetuned_args.json")
    checkpoint_path = os.path.join(checkpoint_dir, "finetuned_500.pth")

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

    model_class = {
        "GNN": GNN,
        "PPGN": PPGN,
        "NGNN": NGNN,
        "GNNAK": GNNAK,
        "IDGNN": IDGNN,
        "I2GNN": I2GNN,
    }.get(args.model)

    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model_class(
        args.dataset,
        num_layers=args.layers,
        edge_attr_dim=1,
        target=args.target,
        y_ndim=2,
    )
    model.load_state_dict(checkpoint)
    model.eval()

    return model, args


def MyTransform(data):
    data.y = data.y[:, int(args.target)]
    return data


def load_dataset(args, dataset_name, split="test"):
    """
    Load and preprocess a dataset.
    """
    def pre_transform(g):
        return create_subgraphs(
            g,
            args.h,
            max_nodes_per_hop=args.max_nodes_per_hop,
            node_label=args.node_label,
            use_rd=args.use_rd,
            save_relabel=True,
        )

    def pre_transform2(g):
        return create_subgraphs2(
            g,
            args.h,
            max_nodes_per_hop=args.max_nodes_per_hop,
            node_label=args.node_label,
            use_rd=args.use_rd,
        )

    processed_name = None

    if args.model == "GNN" or args.model == "PPGN":
        processed_name = "processed"
        my_pre_transform = None
        print("Loading from %s" % "processed")
    elif args.model == "NGNN" or args.model == "GNNAK" or args.model == "IDGNN":
        processed_name = "processed_n_h" + str(args.h) + "_" + args.node_label
        my_pre_transform = pre_transform
    elif args.model == "I2GNN":
        processed_name = "processed_nn_h" + str(args.h) + "_" + args.node_label
        my_pre_transform = pre_transform2
    else:
        print("Error: no such model!")
        exit(1)

    dataset = dp.dataset_random_graph(
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
            y_hat = y_hat * std[args.target] + mean[args.target]

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

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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



if __name__ == "__main__":
    # SET_1 : Mean = 26.825, Std = 155.318
    MODEL = "GNN"
    TARGET = 2
    SET = 0
    # output/data_esc/GNNAK/Checkpoints/0/20241002040818
    # /userhome/matin/Dev/MLSC/output/data_esc/GNN/Checkpoints/0/20241001181525
    # "output/data_esc/I2GNN/Checkpoints/0/20241003062639"
    # output/data_esc/I2GNN/Checkpoints/2/20241003082940
    # output/data_esc/IDGNN/Checkpoints/2/20241005162119
    # output/finetuned_models/GNN/0/Set_1/finetuned_500.pth
    checkpoint_dir = f"/workspace/output/finetuned_models/{MODEL}/{TARGET}/Set_{SET}/"
    print(f"Loading model from checkpoint: {checkpoint_dir}")
    model, args = load_model_from_checkpoint(checkpoint_dir)
    print(f"Model loaded: {type(model).__name__}")

    test_dataset_name = "Set_3"  # Replace with your actual test dataset name
    for test_dataset_name in [f"Set_{SET}"]: # "data_esc", "Set_1", "Set_2", "Set_3",
        print(f"Running test on dataset: {test_dataset_name}")
        test_results = run_test_on_dataset(model, args, test_dataset_name)

        print(f"Test Results:")
        print(f"  MAE: {test_results['mae']}")
        print(f"  Number of samples: {len(test_results['predictions'])}")

        # Optionally, save the results to a file
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"fine_{test_dataset_name}_{MODEL}_{TARGET}_test_results.json")
        with open(output_file, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {output_file}")
