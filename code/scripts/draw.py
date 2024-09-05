import os
import re
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Regular expression patterns
MAE_PATTERN = re.compile(r"Epoch: \d+, LR: [\d.]+, Loss: [\d.]+, Validation MAE: [\d.]+, Test MAE: ([\d.]+)")
COMMAND_PATTERN = re.compile(r"python run_count\.py.*--target (\d+).*--model (\w+)")
MEAN_STD_PATTERN = re.compile(r"Mean = ([\d.]+), Std = ([\d.]+)")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process log files and generate plots.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing the log files")
    parser.add_argument("--output", type=str, default="output", help="Output directory to save the plots (default: output)")
    return parser.parse_args()

def extract_info_from_file(filepath: str) -> Optional[Tuple[int, str, float, float, float]]:
    """
    Extract relevant information from a log file.
    
    Returns:
    Tuple of (target, model, mae, mean, std) if successful, None otherwise.
    """
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            
            command_match = COMMAND_PATTERN.search(content)
            if not command_match:
                logger.warning(f"Could not extract model and target from {filepath}")
                return None
            
            target = int(command_match.group(1))
            model = command_match.group(2)

            mean_std_match = MEAN_STD_PATTERN.search(content)
            if not mean_std_match:
                logger.warning(f"Could not extract Mean and Std from {filepath}")
                mean, std = None, None
            else:
                mean = float(mean_std_match.group(1))
                std = float(mean_std_match.group(2))

            lines = content.splitlines()
            last_line = lines[-1].strip() if not lines[-1].startswith("python") else lines[-2].strip()
            
            mae_match = MAE_PATTERN.search(last_line)
            if not mae_match:
                logger.warning(f"No MAE match found in last line of {filepath}")
                logger.warning(f"Last line: {last_line}")
                return None
            
            mae_str = mae_match.group(1)
            mae = float(mae_str) if mae_str != 'nan' else None
            
            logger.info(f"File: {os.path.basename(filepath)}")
            logger.info(f"Model: {model}, Target: {target}, MAE: {mae}, Mean: {mean}, Std: {std}")
            logger.info("---")
            
            return target, model, mae, mean, std
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None

def process_input_files(input_dir: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Process all input files in the given directory and return the results."""
    results = {}
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    for filename in tqdm(files, desc="Processing files", unit="file"):
        filepath = os.path.join(input_dir, filename)
        info = extract_info_from_file(filepath)
        if info:
            target, model, mae, mean, std = info
            if target not in results:
                results[target] = {}
            results[target][model] = {'mae': mae, 'mean': mean, 'std': std}
    return results

def plot_results(results: Dict[int, Dict[str, Dict[str, float]]], output_dir: str, dataset_name: str):
    """Plot the results and save the charts as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for target, target_data in results.items():
        models = list(target_data.keys())
        if not models:
            logger.warning(f"No data to plot for target {target}")
            continue
        
        maes = [target_data[model]['mae'] for model in models]
        means = [target_data[model]['mean'] for model in models]
        stds = [target_data[model]['std'] for model in models]
        
        plt.figure(figsize=(12, 6))
        
        min_mae_index = maes.index(min(maes))
        colors = ['lightblue' if i != min_mae_index else 'red' for i in range(len(models))]
        
        plt.bar(models, maes, color=colors)
        plt.title(f'Test MAE Comparison for {dataset_name} - Target {target}\nMean = {means[0]:.3f}, Std = {stds[0]:.3f}')
        plt.xlabel('Models')
        plt.ylabel('Test MAE')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(maes):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{dataset_name}_target_{target}_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved plot for {dataset_name} target {target} to {plot_path}")

def main():
    args = parse_arguments()
    
    logger.info(f"Processing input files from: {args.input}")
    logger.info(f"Saving output to: {args.output}")
    
    # Extract dataset name from input directory
    dataset_name = os.path.basename(os.path.normpath(args.input))
    logger.info(f"Dataset name: {dataset_name}")
    
    results = process_input_files(args.input)
    logger.info("Results:")
    logger.info(results)
    
    plot_results(results, args.output, dataset_name)
    logger.info("All charts have been saved as PNG files.")

if __name__ == "__main__":
    main()