import os
import sys
import json
import torch
import docker
import logging
import argparse
import subprocess
from rwdq import run_query

class ColoredLogger:
    # ANSI escape codes for different colors (foreground and background)
    COLORS = {
        'DEBUG': '\033[1;36;44m',    # Bold Cyan text on Blue background
        'INFO': '\033[1;30;43m',     # Bold Black text on Yellow background
        'WARNING': '\033[1;37;41m',  # Bold White text on Red background
        'ERROR': '\033[1;37;45m',    # Bold White text on Magenta background
        'CRITICAL': '\033[1;37;42m', # Bold White text on Green background
    }
    RESET = '\033[0m'

    def __init__(self, name='ColoredLogger', level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self.ColoredFormatter(self.COLORS, self.RESET))
        self.logger.addHandler(handler)

    class ColoredFormatter(logging.Formatter):
        def __init__(self, colors, reset):
            super().__init__('%(asctime)s - %(levelname)s - %(message)s')
            self.colors = colors
            self.reset = reset

        def format(self, record):
            color = self.colors.get(record.levelname, self.reset)
            record.msg = f"{color}{record.msg}{self.reset}"
            return f"{color}{super().format(record)}{self.reset}"

    def debug(self, message): self.logger.debug(message)
    def info(self, message): self.logger.info(message)
    def warning(self, message): self.logger.warning(message)
    def error(self, message): self.logger.error(message)
    def critical(self, message): self.logger.critical(message)

logger = ColoredLogger()

def sample_oracle(sample_config):
    for key, value in sample_config['constraints'].items():
        if key.startswith('Set_'):
            output_folder = os.path.join(sample_config['output_folder'], key)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            config_file = os.path.join(output_folder, 'config.json')
            with open(config_file, 'w') as file:
                json.dump(value, file, indent=4)

            samples = run_query(config_file, sample_config['db_path'], sample_config['save_path'])
            split_ratio = [int(x) for x in sample_config['split_ratio'].split(':')]
            s1, s2 = [int(len(samples) * r / sum(split_ratio)) for r in split_ratio[:2]]
            dataset = {
                'train': samples[:s1],
                'val': samples[s1:s1+s2],
                'test': samples[s1+s2:]
            }
            torch.save(dataset, os.path.join(output_folder, 'dataset.pt'))

def plot_sampled(plots_path):
    pass

def run_experiments(algorithms):
    for alg in algorithms[1:4]:
        logger.info(f"Running experiments for {alg['name']}")
        client = docker.from_env()
        container_config = alg.get('container_config', {})
        try:
            client.containers.get(container_config['name']).remove(force=True)
        except docker.errors.NotFound:
            pass
        container = client.containers.run(**container_config)
        try:
            subprocess.run(alg.get('command'))
        finally:
            # container.stop()
            container.remove(force=True)

def plot_results(plots_path):
    pass


def main(config_path, **kwargs):
    # load the configuration file
    config = json.load(open(config_path, 'r'))
    control = config.get('execution_control', {})

    # 1. Sample the oracle dataset
    if control.get('sample_oracle', True):
        sample_oracle(config['sample_config'])
        logger.critical(f"Oracle sampling completed. Datasets saved to {config['sample_config']['output_folder']}")
    else:
        logger.critical("Oracle sampling skipped.")

    # 2. analyze the sampled data and plot
    if control.get('plot_sampled', True):
        plot_sampled(config['plots_path'])
        logger.critical("Plotting sampled dataset completed.")
    else:
        logger.critical("Plotting sampled dataset skipped.")

    # 3. run experiments in containers
    if control.get('run_experiments', True):
        run_experiments(config['algorithms'])
        logger.critical("Experiments completed.")
    else:
        logger.critical("Experiments skipped.")

    # 4. analyze the results and plot
    if control.get('plot_results', True):
        plot_results(config['plots_path'])
        logger.critical("Plotting results completed.")
    else:
        logger.critical("Plotting results skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running experiments")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)