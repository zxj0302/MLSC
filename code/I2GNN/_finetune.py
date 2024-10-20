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

MODELS = ["GNN", "GNNAK"] # "GNN", "GNNAK", , "PPGN", "IDGNN"
NUM_EPOCHS = 300
NUM_TARGETS = 11
BATCH_SIZE = {
    "GNN":[256, 256, 256, 256],
    "PPGN":[1, 2, 2, 2],
    "GNNAK":[2, 4, 4, 8],
    "IDGNN":[2, 4, 4, 8],
    "I2GNN":[1, 2, 4, 4]
    }
TARGETS = [13]


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
        print(f"Running dataset_rg on {dataset_name} split {split}...")
        dataset = dp.dataset_random_graph(
            dataname=dataset_name,
            processed_name=processed_name,
            transform=my_transform,
            pre_transform=my_pre_transform,
            split=split,
        )
        print(f"Finished dataset_rg on dataset {dataset_name} split {split}")
        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")


def finetune_model(model, args, train_dataset, val_dataset, device, num_epochs=200, dataset_name="Set_00"):
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
            train_dataset, batch_size=BATCH_SIZE[args.model][-args.h], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE[args.model][-args.h])

        best_val_error = None
        for epoch in tqdm(range(1, num_epochs+1), desc="Finetuning"):
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
            if epoch % 25 == 0:
                logger.info(
                    f"Epoch: {epoch:03d}, LR: {lr:.7f}, Loss: {avg_loss:.7f}, Validation MAE: {val_error:.7f}"
                )
            if epoch <= NUM_EPOCHS:
                save_model(model, args, f"/workspace/output/final_fine/{dataset_name}/{args.model}/{args.target}", args.target, epoch)

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


def save_model(model, args, save_dir, target, epoch=500):
    """
    Save the finetuned model.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"finetuned_{epoch}.pth")
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
    base_save_dir = "/workspace/output/final_fine/"
    os.makedirs(base_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")

    print(f"Using device: {device}")

    for mod in MODELS:
        # if mod == "PPGN":
        #     BATCH_SIZE = 4
        # else:
        #     BATCH_SIZE = 32
        base_checkpoint_dir = f"/workspace/output/data_esc/{mod}/Checkpoints"
        for finetune_dataset_name in ["Set_2", "Set_3", "Set_5"]:
            # finetune_dataset_name = "Set_1"  # Replace with your finetuning dataset name
            for target in TARGETS:  # Assuming 29 targets as in the original code
                print(f"Processing target {target} with model {mod} on dataset {finetune_dataset_name}")
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

                    # y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
                    # mean = y_train_val.mean(dim=0)
                    # std = y_train_val.std(dim=0)
                    # train_dataset.data.y = (train_dataset.data.y - mean) / std
                    # val_dataset.data.y = (val_dataset.data.y - mean) / std
                    # test_dataset.data.y = (test_dataset.data.y - mean) / std
                    # print("Mean = %.3f, Std = %.3f" % (mean[args.target], std[args.target]))

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
                    save_dir = os.path.join(
                        base_save_dir, finetune_dataset_name, str(mod), str(target)
                    )
                    save_model(finetuned_model, args, save_dir, target, epoch=NUM_EPOCHS)

                except Exception as e:
                    logger.error(f"Error processing target {target}: {str(e)}")
                    continue

def dataset_load_time():
    TARGET_DIAM = [2, 1,  3, 2, 2, 2, 2, 1,   4, 3, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    import time
    args_path = "/workspace/output/data_esc/GNN/Checkpoints/0/20241010212009/args.json"
    with open(args_path, "r") as f:
        saved_args = json.load(f)

    # Create args object and populate it
    args = type("Args", (), {})()
    for key, value in saved_args.items():
        setattr(args, key, value)

    res = json.load(open("/workspace/output/load_time.json", "r"))
    for dataset in [f"Set_{i}" for i in range(6, 11)]:
        res[dataset] = {} if dataset not in res else res[dataset]
        for alg in ["GNN", "IDGNN"]:
            res[dataset][alg] = {} if alg not in res[dataset] else res[dataset][alg]
            for target in [0,1,2,8]:
                res[dataset][alg][target] = {} if target not in res[dataset][alg] else res[dataset][alg][target]
                # update args
                args.dataset = dataset
                args.model = alg
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
