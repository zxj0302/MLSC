import subprocess
import time
import resource
import json
import os
from typing import Dict, Any, List, Tuple
import logging
import itertools
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, 'r') as f:
        return json.load(f)

def generate_esc_commands(config: Dict[str, Any]) -> List[str]:
    base_command = "python run_graphcount.py"
    param_combinations = list(itertools.product(
        config['batch_size'], config['target'], config['model'],
        config['h'], config['lr'], config['layers'], config['dataset']
    ))
    return [f"{base_command} --batch_size {p[0]} --target {p[1]} --model {p[2]} --h {p[3]} --lr {p[4]} --layers {p[5]} --dataset {p[6]}" for p in param_combinations]

def generate_lpp_commands(config: Dict[str, Any]) -> List[str]:
    base_command = "python main.py"
    param_combinations = list(itertools.product(
        config['dataset'], config['pattern'], config['model'],
        config['epoch'], config['batch_size'], config['device']
    ))
    return [f"{base_command} --dataset {p[0]} --pattern {p[1]} --model {p[2]} --epoch {p[3]} --batch_size {p[4]} --device {p[5]}" for p in param_combinations]

def generate_commands(config: Dict[str, Any], family: str) -> List[str]:
    if family == "ESC":
        return generate_esc_commands(config)
    elif family == "LPP":
        return generate_lpp_commands(config)
    else:
        raise ValueError(f"Unsupported algorithm family: {family}")

def extract_command_info(command: str, family: str) -> Tuple[str, str]:
    if family == "ESC":
        target = re.search(r'--target (\d+)', command).group(1)
        dataset = re.search(r'--dataset (\S+)', command).group(1)
    elif family == "LPP":
        target = re.search(r'--pattern (\S+)', command).group(1)
        dataset = re.search(r'--dataset (\S+)', command).group(1)
    else:
        raise ValueError(f"Unsupported algorithm family: {family}")
    return target, dataset

def create_log_file(dataset: str, target: str, family: str) -> str:
    timestamp = time.strftime('%Y%m%d%H%M%S')
    log_filename = f"{family}_{dataset}_t{target}_{timestamp}.log"
    log_file = os.path.join('logs', family, dataset, log_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    return log_file

def execute_command(full_command: str) -> Tuple[float, resource.struct_rusage, resource.struct_rusage, int]:
    start_time = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()
    end_time = time.time()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    return end_time - start_time, usage_start, usage_end, process.returncode

def process_log_file(log_file: str, family: str) -> None:
    if family == "ESC":
        process_esc_log_file(log_file)
    elif family == "LPP":
        process_lpp_log_file(log_file)
    else:
        raise ValueError(f"Unsupported algorithm family: {family}")

def process_esc_log_file(log_file: str) -> None:
    with open(log_file, 'r') as f:
        log_content = f.readlines()
    
    initial_info = extract_esc_initial_info(log_content)
    epoch_lines = extract_esc_epoch_lines(log_content)
    
    filtered_content = initial_info + [''] + epoch_lines
    
    with open(log_file, 'w') as f:
        f.write('\n'.join(filtered_content))

def process_lpp_log_file(log_file: str) -> None:
    # Placeholder function for LPP log processing
    logger.info(f"Processing LPP log file: {log_file}")
    # We'll implement this function later when we have a sample LPP log file

def extract_esc_initial_info(log_content: List[str]) -> List[str]:
    initial_info = []
    for line in log_content:
        if "Command line input:" in line or "Using" in line or "Mean =" in line:
            initial_info.append(line.strip())
        if len(initial_info) == 3:
            break
    return initial_info

def extract_esc_epoch_lines(log_content: List[str]) -> List[str]:
    epoch_lines = []
    for i, line in enumerate(log_content):
        if line.startswith("Epoch:") and ", LR:" in line:
            epoch_info = line.strip()
            if i + 1 < len(log_content) and "Test MAE norm:" in log_content[i + 1]:
                epoch_info += " " + log_content[i + 1].strip()
            epoch_lines.append(epoch_info)
    return epoch_lines

def run_command(command: str, family: str) -> Dict[str, Any]:
    target, dataset = extract_command_info(command, family)
    log_file = create_log_file(dataset, target, family)
    
    full_command = f"{command} > {log_file} 2>&1"
    logger.info(f'Running command with log: {full_command}')

    time_taken, usage_start, usage_end, return_code = execute_command(full_command)
    
    process_log_file(log_file, family)
    
    return {
        'command': command,
        'log_file': log_file,
        'return_code': return_code,
        'time_taken': time_taken,
        'user_time': usage_end.ru_utime - usage_start.ru_utime,
        'system_time': usage_end.ru_stime - usage_start.ru_stime,
        'max_memory': usage_end.ru_maxrss,
        'date': time.strftime('%Y-%m-%d %H:%M:%S')
    }

def save_result(result: Dict[str, Any], index: int, family: str) -> None:
    result_filename = f'results/{family}/experiment_{index}.json'
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    with open(result_filename, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'Results saved to {result_filename}')

def run_experiments(config: Dict[str, Any], family: str) -> None:
    commands = generate_commands(config, family)
    
    for i, command in enumerate(commands, 1):
        logger.info(f'Running command {i}/{len(commands)}: {command}')
        result = run_command(command, family)
        save_result(result, i, family)

def main():
    families = ["ESC", "LPP"]
    for family in families:
        config_file = f'{family}_config.json'
        config = read_config(config_file)
        logger.info(f"Running experiments for {family}")
        run_experiments(config, family)

if __name__ == "__main__":
    main()