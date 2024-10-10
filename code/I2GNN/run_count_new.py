import argparse
from argparse import Namespace
import random
import numpy as np
import os
import time
import torch
from torch.nn import L1Loss
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import data_processing as dp
from typing import Union, Dict, List, Tuple
from count_models import GNN, PPGN, NGNN, GNNAK, IDGNN, I2GNN
from tqdm import tqdm
import json, sys


def MyTransform(data: Data, args: argparse.Namespace) -> Data:
    """
    Transform the data by selecting a specific target.

    Args:
        data (Data): The input data.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Data: The transformed data.
    """
    data.y = data.y[:, int(args.target)]
    return data


def setup_environment(args: argparse.Namespace) -> None:
    """
    Set up the environment for the experiment.
    
    This function sets random seeds for reproducibility,
    determines the device (CPU or CUDA) to use,
    and sets up the results directory.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Determine the device to use
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Set up results directory
    if args.save_appendix == "":
        args.save_appendix = "_" + time.strftime("%Y%m%d%H%M%S")
    args.res_dir = os.path.join("results", f"{args.dataset}_{args.model}{args.save_appendix}")
    
    print(f"Results will be saved in {args.res_dir}")
    os.makedirs(args.res_dir, exist_ok=True)

    # Save command line input
    cmd_input = "python " + " ".join(sys.argv) + "\n"
    with open(os.path.join(args.res_dir, "cmd_input.txt"), "a") as f:
        f.write(cmd_input)
    print(f"Command line input: {cmd_input} is saved.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="I2GNN for counting experiments.")

    # General settings
    parser.add_argument(
        "--target",
        default=0,
        type=int,
        help="0 for detection of tri-cycle, 3,4,...,8 for counting of cycles",
    )
    parser.add_argument(
        "--ab", action="store_true", default=False, help="Ablation study flag"
    )

    # Base GNN settings
    parser.add_argument("--model", type=str, default="GNN", help="Model type")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")

    # Nested GNN settings
    parser.add_argument(
        "--h",
        type=int,
        default=None,
        help="Hop of enclosing subgraph; if None, will not use NestedGNN",
    )
    parser.add_argument("--max_nodes_per_hop", type=int, default=None)
    parser.add_argument(
        "--node_label",
        type=str,
        default="hop",
        help="Apply distance encoding to nodes within each subgraph, use node labels as additional node features; "
        'support "hop", "drnl", "spd". For "spd", you can specify number of spd to keep by "spd3", "spd4", "spd5", etc. Default "spd"=="spd2".',
    )
    parser.add_argument(
        "--use_rd",
        action="store_true",
        default=False,
        help="Use resistance distance as additional node labels",
    )

    # Training settings
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lr_decay_factor", type=float, default=0.9, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for learning rate scheduler"
    )

    # Other settings
    parser.add_argument("--seed", type=int, default=233, help="Random seed")
    parser.add_argument(
        "--save_appendix",
        default="",
        help="What to append to save-names when saving results",
    )
    parser.add_argument(
        "--keep_old",
        action="store_true",
        default=False,
        help="If True, do not overwrite old .py files in the result folder",
    )
    parser.add_argument("--dataset", default="count_cycle", help="Dataset name")
    parser.add_argument(
        "--load_model", default=None, help="Path to load a pre-trained model"
    )
    parser.add_argument("--eval", default=0, type=int, help="Evaluation flag")
    parser.add_argument("--train_only", default=0, type=int, help="Train only flag")

    return parser.parse_args()


def load_datasets(args: argparse.Namespace) -> Tuple[Data, Data, Data]:
    """
    Load and preprocess the datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple[Data, Data, Data]: Tuple containing train, validation, and test datasets.
    """
    def get_pre_transform() -> Union[callable, None]:
        if args.h is None:
            return None
        
        from utils import create_subgraphs, create_subgraphs2

        if args.model in ["NGNN", "GNNAK", "IDGNN"]:
            return lambda g: create_subgraphs(
                g, args.h, max_nodes_per_hop=args.max_nodes_per_hop,
                node_label=args.node_label, use_rd=args.use_rd, save_relabel=True
            )
        elif args.model == "I2GNN":
            return lambda g: create_subgraphs2(
                g, args.h, max_nodes_per_hop=args.max_nodes_per_hop,
                node_label=args.node_label, use_rd=args.use_rd
            )
        else:
            return None

    pre_transform = get_pre_transform()

    processed_name = (
        "processed" if args.model in ["GNN", "PPGN"] else
        f"processed_n_h{args.h}_{args.node_label}" if args.model in ["NGNN", "GNNAK", "IDGNN"] else
        f"processed_nn_h{args.h}_{args.node_label}" if args.model == "I2GNN" else
        None
    )
    if processed_name is None:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.use_rd:
        processed_name += "_rd"

    # Load datasets
    train_dataset = dp.dataset_random_graph(
        dataname=args.dataset,
        processed_name=processed_name,
        transform=lambda data: MyTransform(data, args),
        pre_transform=pre_transform,
        split="train"
    )
    val_dataset = dp.dataset_random_graph(
        dataname=args.dataset,
        processed_name=processed_name,
        transform=lambda data: MyTransform(data, args),
        pre_transform=pre_transform,
        split="val"
    )
    test_dataset = dp.dataset_random_graph(
        dataname=args.dataset,
        processed_name=processed_name,
        transform=lambda data: MyTransform(data, args),
        pre_transform=pre_transform,
        split="test"
    )

    # Normalize target
    y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
    mean = y_train_val.mean(dim=0)
    std = y_train_val.std(dim=0)

    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.data.y = (dataset.data.y - mean) / std

    print(f"Mean = {mean[args.target]:.3f}, Std = {std[args.target]:.3f}")

    # Store mean and std in args for later use
    args.mean = mean
    args.std = std

    # Ablation study for I2GNN
    if args.ab:
        for dataset in [train_dataset, val_dataset, test_dataset]:
            dataset.data.z[:, 2:] = torch.zeros_like(dataset.data.z[:, 2:])

    return train_dataset, val_dataset, test_dataset

def create_model(args: argparse.Namespace, train_dataset: Data) -> Union[GNN, PPGN, NGNN, GNNAK, IDGNN, I2GNN]:
    """
    Initialize the model based on the provided arguments and training dataset.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_dataset (Data): The training dataset.

    Returns:
        Union[GNN, PPGN, NGNN, GNNAK, IDGNN, I2GNN]: The initialized model.

    Raises:
        ValueError: If an unsupported model type is specified.
    """
    model_classes = {
        "GNN": GNN,
        "PPGN": PPGN,
        "NGNN": NGNN,
        "GNNAK": GNNAK,
        "IDGNN": IDGNN,
        "I2GNN": I2GNN
    }

    if args.model not in model_classes:
        raise ValueError(f"Unsupported model type: {args.model}")

    model_class = model_classes[args.model]

    kwargs = {
        "num_layers": args.layers,
        "edge_attr_dim": 1,
        "target": args.target,
        "y_ndim": 2,
    }

    model = model_class(train_dataset, **kwargs)
    print(f"Using {model.__class__.__name__} model")

    if args.load_model:
        try:
            checkpoint = torch.load(args.load_model, map_location=args.device)
            model.load_state_dict(checkpoint)
            print(f"Loaded pre-trained model from {args.load_model}")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")

    return model.to(args.device)

def train(
    model: torch.nn.Module,
    train_loader: Union[DataLoader, Dict[str, DataLoader]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        device (torch.device): The device to use for training.

    Returns:
        float: The average loss for this epoch.
    """
    model.train()
    total_loss = 0
    total_graphs = 0

    for data in train_loader:
        optimizer.zero_grad()

        if isinstance(data, dict):
            # For nested GNN models
            data = {key: value.to(device) for key, value in data.items()}
            num_graphs = data[list(data.keys())[0]].num_graphs
        else:
            # For regular GNN models
            data = data.to(device)
            num_graphs = data.num_graphs

        y = data.y.view([-1, 1])
        y_pred = model(data)

        loss_fn = L1Loss()
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_graphs += num_graphs

    average_loss = total_loss / total_graphs
    return average_loss


def test(
    model: torch.nn.Module,
    loader: Union[DataLoader, Dict[str, DataLoader]],
    args: argparse.Namespace,
    output: bool = False,
) -> Union[float, Tuple[List[float], List[float]]]:
    """
    Evaluate the model on the given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader for evaluation data.
        args (argparse.Namespace): Parsed command-line arguments.
        output (bool, optional): If True, return predictions and ground truth. Defaults to False.

    Returns:
        Union[float, Tuple[List[float], List[float]]]:
            If output is False, returns the mean absolute error.
            If output is True, returns a tuple of (predictions, ground truth).
    """
    model.eval()
    total_error = 0
    total_samples = 0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for data in loader:
            if isinstance(data, dict):
                # For nested GNN models
                data = {key: value.to(args.device) for key, value in data.items()}
                y = data[list(data.keys())[0]].y
            else:
                # For regular GNN models
                data = data.to(args.device)
                y = data.y

            y_pred = model(data)[:, 0]

            error = torch.sum(torch.abs(y_pred - y))
            total_error += error.item()
            total_samples += y.size(0)

            if output:
                # Denormalize predictions and ground truth
                predictions.extend(
                    (y_pred * args.std[args.target] + args.mean[args.target])
                    .cpu()
                    .numpy()
                    .tolist()
                )
                ground_truth.extend(
                    (y * args.std[args.target] + args.mean[args.target])
                    .cpu()
                    .numpy()
                    .tolist()
                )

    mean_absolute_error = total_error / total_samples * args.std[args.target]

    if output:
        return predictions, ground_truth
    else:
        return mean_absolute_error


def loop(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
) -> dict:
    """
    Main training and evaluation loop.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: A dictionary containing the training results and time profiling.
    """
    best_val_error = float("inf")
    patience_counter = 0
    times = {}
    start_time = time.time()

    # Create a directory for checkpoints
    timestamp = time.strftime("%Y%m%d%H%M%S")
    checkpoint_dir = f"/workspace/output/{args.dataset}/{args.model}/Checkpoints/{args.target}/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    pbar = tqdm(range(1, args.epochs + 1), desc="Training")
    for epoch in pbar:
        # Training
        train_start = time.time()
        train_loss = train(model, train_loader, optimizer, args.device)
        times["training"] = time.time() - train_start

        # Validation
        val_start = time.time()
        val_error = test(model, val_loader, args)
        times["validation"] = time.time() - val_start

        # Update learning rate
        scheduler.step(val_error)

        # Check for improvement
        if val_error <= best_val_error:
            best_val_error = val_error
            patience_counter = 0

            # Save the best model
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth")
            )
        else:
            patience_counter += 0

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

        # Update progress bar
        pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val MAE": f"{val_error:.4f}",
                "LR": f'{optimizer.param_groups[0]["lr"]:.6f}',
            }
        )

        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_error": best_val_error,
                },
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))

    # Final evaluation on test set
    test_start = time.time()
    test_error = test(model, test_loader, args)
    times["testing"] = time.time() - test_start

    # Get predictions and ground truth for further analysis
    y_pred, y_true = test(model, test_loader, args, output=True)

    times["total"] = time.time() - start_time

    results = {
        "predictions": y_pred,
        "ground_truth": y_true,
        "time_profile": times,
    }

    return results


def save_results(args: argparse.Namespace, results: dict):
    """
    Save results to a JSON file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        results (dict): Dictionary containing the results to save.
    """
    output_dir = os.path.join("/workspace/output", args.dataset, args.model)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"target_{args.target}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


def main():
    args = parse_arguments()
    setup_environment(args)
    train_dataset, val_dataset, test_dataset = load_datasets(args)
    model = create_model(args, train_dataset)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.patience,
        min_lr=0.00001,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    results = loop(
        model, train_loader, val_loader, test_loader, optimizer, scheduler, args
    )
    save_results(args, results)
    # Rest of your main code here, using args as needed


if __name__ == "__main__":
    main()
