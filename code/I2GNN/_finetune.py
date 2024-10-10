import os
import json
import torch
import numpy as np
from dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import model definitions
from count_models import GNN, PPGN, NGNN, GNNAK, IDGNN, I2GNN
import data_processing as dp
from utils import create_subgraphs, create_subgraphs2

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS = ["GNNAK", "I2GNN", "PPGN", "IDGNN"] # "GNN", "GNNAK", "I2GNN", "PPGN", "IDGNN"
NUM_EPOCHS = 500
NUM_TARGETS = 11
BATCH_SIZE = 16


def load_model_from_checkpoint(checkpoint_dir):
    """
    Load a model from a checkpoint directory containing args.json and the latest checkpoint file.
    """
    try:
        args_path = os.path.join(checkpoint_dir, "args.json")
        if not os.path.exists(args_path):
            raise FileNotFoundError(
                f"args.json not found in the checkpoint directory: {args_path}"
            )

        with open(args_path, "r") as f:
            saved_args = json.load(f)

        # Create args object and populate it
        args = type("Args", (), {})()
        for key, value in saved_args.items():
            setattr(args, key, value)

        # Find the latest checkpoint file
        checkpoint_files = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("cpt_") and f.endswith(".pth")
        ]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        latest_checkpoint = max(
            checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

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
        # print(vars(args))

        return model, args

    except Exception as e:
        logger.error(f"Error loading model from checkpoint: {str(e)}")
        raise


def load_dataset(args, dataset_name, split="train"):
    """
    Load and preprocess a dataset.
    """
    try:

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

        if args.model in ["GNN", "PPGN"]:
            processed_name, my_pre_transform = "processed", None
        elif args.model in ["NGNN", "GNNAK", "IDGNN"]:
            processed_name = f"processed_n_h{args.h}_{args.node_label}"
            my_pre_transform = pre_transform
        elif args.model == "I2GNN":
            processed_name = f"processed_nn_h{args.h}_{args.node_label}"
            my_pre_transform = pre_transform2
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        def my_transform(data):
            data.y = data.y[:, args.target]
            return data

        dataset = dp.dataset_random_graph(
            dataname=dataset_name,
            processed_name=processed_name,
            transform=my_transform,
            pre_transform=my_pre_transform,
            split=split,
        )
        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def finetune_model(model, args, train_dataset, val_dataset, device, num_epochs=200):
    """
    Finetune the model on the given dataset with validation.
    """
    try:
        model.train()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.patience,
            min_lr=0.00001,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        best_val_error = None
        for epoch in tqdm(range(num_epochs), desc="Finetuning"):
            # Training
            model.train()
            total_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                if isinstance(data, dict):
                    data = {k: v.to(device) for k, v in data.items()}
                else:
                    data = data.to(device)

                out = model(data)
                y = data.y.view([data.y.size(0), 1])
                loss = F.l1_loss(out, y)
                loss.backward()
                total_loss += loss.item() * y.size(0)
                optimizer.step()

            avg_loss = total_loss / len(train_dataset)

            # Validation
            val_error = test(val_loader, model, args, device)
            scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                best_val_error = val_error
                # You might want to save the best model here

            # Logging
            lr = scheduler.optimizer.param_groups[0]["lr"]
            if (epoch + 1) % 25 == 0:
                logger.info(
                    f"Epoch: {epoch+1:03d}, LR: {lr:.7f}, Loss: {avg_loss:.7f}, Validation MAE: {val_error:.7f}"
                )

        return model

    except Exception as e:
        logger.error(f"Error during finetuning: {str(e)}")
        raise


def test(loader, model, args, device):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            if isinstance(data, dict):
                data = {k: v.to(device) for k, v in data.items()}
            else:
                data = data.to(device)
            out = model(data)
            y = data.y.view([data.y.size(0), 1])
            error += torch.sum(torch.abs(out - y)).item()
    return error / len(loader.dataset)


def save_model(model, args, save_dir, target):
    """
    Save the finetuned model.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"finetuned_{NUM_EPOCHS}.pth")
        torch.save(model.state_dict(), save_path)

        # Save updated args
        args_path = os.path.join(save_dir, f"finetuned_args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info(f"Finetuned model and args saved for target {target}")

    except Exception as e:
        logger.error(f"Error saving finetuned model: {str(e)}")
        raise


def find_latest_timestamp_folder(base_path):
    """
    Searches for the latest timestamp folder in the given directory.

    Args:
    base_path (str): The path to the directory containing timestamp subfolders.

    Returns:
    str: The name of the latest timestamp subfolder, or None if no valid subfolders are found.

    Raises:
    FileNotFoundError: If the base_path does not exist.
    """
    try:
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The directory {base_path} does not exist.")

        # List all subfolders in the given directory
        subfolders = [
            f
            for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
        ]

        # Filter and parse valid timestamp folders
        valid_timestamps = []
        for folder in subfolders:
            try:
                # Attempt to parse the folder name as a timestamp
                timestamp = datetime.strptime(folder, "%Y%m%d%H%M%S")
                valid_timestamps.append((timestamp, folder))
            except ValueError:
                # If parsing fails, it's not a valid timestamp folder, so we skip it
                continue

        if not valid_timestamps:
            return None

        # Sort the valid timestamps and return the name of the latest one
        latest_timestamp = max(valid_timestamps, key=lambda x: x[0])
        return latest_timestamp[1]

    except Exception as e:
        print(f"An error occurred while finding the latest timestamp folder: {str(e)}")
        return None


def main():
    base_save_dir = "/workspace/output/finetuned_models/"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")

    print(f"Using device: {device}")

    for mod in MODELS:
        base_checkpoint_dir = f"/workspace/output/data_esc/{mod}/Checkpoints"
        for finetune_dataset_name in [f"Set_{i}" for i in range(0, 8)][:1]:
            # finetune_dataset_name = "Set_1"  # Replace with your finetuning dataset name
            for target in range(
                NUM_TARGETS
            ):  # Assuming 29 targets as in the original code
                try:
                    target_checkpoint_dir = os.path.join(
                        base_checkpoint_dir, str(target)
                    )
                    latest_timestamp = find_latest_timestamp_folder(
                        target_checkpoint_dir
                    )

                    if latest_timestamp is None:
                        logger.warning(
                            f"No valid checkpoint folder found for target {target}. Skipping."
                        )
                        continue

                    checkpoint_dir = os.path.join(
                        target_checkpoint_dir, latest_timestamp
                    )
                    logger.info(
                        f"Processing target {target} with checkpoint from {latest_timestamp}"
                    )

                    # Load model and args
                    model, args = load_model_from_checkpoint(checkpoint_dir)
                    args.target = target  # Update target in args

                    # Load finetuning datasets
                    train_dataset = load_dataset(
                        args, finetune_dataset_name, split="train"
                    )
                    val_dataset = load_dataset(args, finetune_dataset_name, split="val")
                    # test_dataset = load_dataset(args, finetune_dataset_name, split="test")

                    y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
                    mean = y_train_val.mean(dim=0)
                    std = y_train_val.std(dim=0)
                    train_dataset.data.y = (train_dataset.data.y - mean) / std
                    val_dataset.data.y = (val_dataset.data.y - mean) / std
                    # test_dataset.data.y = (test_dataset.data.y - mean) / std
                    print("Mean = %.3f, Std = %.3f" % (mean[args.target], std[args.target]))

                    # Finetune model
                    finetuned_model = finetune_model(
                        model,
                        args,
                        train_dataset,
                        val_dataset,
                        device,
                        num_epochs=NUM_EPOCHS,
                    )

                    # Save finetuned model
                    save_dir = os.path.join(
                        base_save_dir, str(mod), str(target), finetune_dataset_name
                    )
                    save_model(finetuned_model, args, save_dir, target)

                except Exception as e:
                    logger.error(f"Error processing target {target}: {str(e)}")
                    continue


if __name__ == "__main__":
    main()
