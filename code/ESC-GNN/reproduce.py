import subprocess
import time
import datetime
import resource
import json
import os
from typing import Dict, Any, List
import logging
import yaml
import itertools
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_config(config_file: str) -> Dict[str, Any]:
    """
    Read the configuration file.
    
    Args:
        config_file (str): Path to the configuration file.
    
    Returns:
        Dict[str, Any]: Configuration parameters.
    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def generate_commands(config: Dict[str, Any]) -> List[str]:
    """
    Generate commands based on the configuration.
    
    Args:
        config (Dict[str, Any]): Configuration parameters.
    
    Returns:
        List[str]: List of commands to be executed.
    """
    base_command = "python run_graphcount.py"
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        config['batch_size'],
        config['target'],
        config['model'],
        config['h'],
        config['lr'],
        config['layers'],
        config['dataset']
    ))
    
    commands = []
    for params in param_combinations:
        command = f"{base_command} --batch_size {params[0]} --target {params[1]} --model {params[2]} --h {params[3]} --lr {params[4]} --layers {params[5]} --dataset {params[6]}"
        commands.append(command)
    
    return commands

def run_command(command: str) -> Dict[str, Any]:
    """
    Execute a bash command and return the output along with resource usage statistics.
    
    Args:
        command (str): The command to be executed.
    
    Returns:
        Dict[str, Any]: A dictionary containing command execution details and resource usage.
    """
    # Extract relevant information from the command
    target = re.search(r'--target (\d+)', command).group(1)
    dataset = re.search(r'--dataset (\S+)', command).group(1)

    # Create a log filename with only dataset and target
    timestamp = time.strftime('%Y%m%d%H%M%S')
    log_filename = f"{dataset}_t{target}_{timestamp}.log"
    log_file = os.path.join('logs', dataset, log_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    full_command = f"{command} > {log_file} 2>&1"
    logger.info(f'Running command with log: {full_command}')

    start_time = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    end_time = time.time()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)

    # Process the log file to extract only necessary information
    with open(log_file, 'r') as f:
        log_content = f.readlines()

    # Extract important initial information
    initial_info = []
    for line in log_content:
        if "Command line input:" in line or "Using" in line or "Mean =" in line:
            initial_info.append(line.strip())
        if len(initial_info) == 3:  # We've found all the information we need
            break

    # Extract only the lines containing epoch information, including Test MAE norm
    epoch_lines = []
    for i, line in enumerate(log_content):
        if line.startswith("Epoch:") and ", LR:" in line:
            epoch_info = line.strip()
            if i + 1 < len(log_content) and "Test MAE norm:" in log_content[i + 1]:
                epoch_info += " " + log_content[i + 1].strip()
            epoch_lines.append(epoch_info)

    # Combine initial info and epoch lines
    filtered_content = initial_info + [''] + epoch_lines  # Add an empty line for separation

    # Write the filtered content back to the log file
    with open(log_file, 'w') as f:
        f.write('\n'.join(filtered_content))
    
    return {
        'command': command,
        'log_file': log_file,
        'return_code': process.returncode,
        'time_taken': end_time - start_time,
        'user_time': usage_end.ru_utime - usage_start.ru_utime,
        'system_time': usage_end.ru_stime - usage_start.ru_stime,
        'max_memory': usage_end.ru_maxrss,
        'date': time.strftime('%Y-%m-%d %H:%M:%S')
    }

def run_experiments(config: Dict[str, Any]) -> None:
    """
    Generate commands from config and execute them, saving the results.
    
    Args:
        config (Dict[str, Any]): Configuration parameters.
    """
    commands = generate_commands(config)
    
    for i, command in enumerate(commands, 1):
        logger.info(f'Running command {i}/{len(commands)}: {command}')
        result = run_command(command)

        result_filename = f'res_matin/experiment_{i}.json'
        os.makedirs(os.path.dirname(result_filename), exist_ok=True)
        
        with open(result_filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f'Results saved to {result_filename}')

def main():
    config_file = 'experiment_config.yaml'
    config = read_config(config_file)
    run_experiments(config)

if __name__ == "__main__":
    main()