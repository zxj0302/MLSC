import os
import sys
import json
import math
import torch
import docker
import logging
import warnings
import argparse
import itertools
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from rwdq import run_query
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

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

def collate_test(data_list):
    edge_index = []
    cumulative_nodes = 0
    for data in data_list:
        edge_index.append(data.edge_index + torch.tensor(cumulative_nodes, dtype=torch.int64))
        cumulative_nodes += data.num_nodes
    
    gt_induced_le5 = torch.cat([data.gt_induced_le5 for data in data_list], dim=0)
    gt_induced_le5_desco = torch.cat([data.gt_induced_le5_desco for data in data_list], dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    
    return Data(edge_index=edge_index, gt_induced_le5=gt_induced_le5, gt_induced_le5_desco=gt_induced_le5_desco, num_nodes=cumulative_nodes, source='combined')

def sample_oracle(sample_config):
    for key, value in sample_config['datasets'].items():
        if key.startswith('Set_'):
            output_folder = os.path.join(sample_config['output_folder'], key)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            config_file = os.path.join(output_folder, 'config.json')
            with open(config_file, 'w') as file:
                json.dump(value["constraints"], file, indent=4)

            samples = run_query(config_file, sample_config['db_path'], sample_config['save_path'])
            split_ratio = [int(x) for x in value['split_ratio'].split(':')]
            s1, s2 = [int(len(samples) * r / sum(split_ratio)) for r in split_ratio[:2]]
            dataset = {
                'train': samples[:s1],
                'val': samples[s1:s1+s2],
                'test': samples[s1+s2:]
            }
            if value['merge_test']:
                dataset['test'] = [collate_test(dataset['test'])]
            torch.save(dataset, os.path.join(output_folder, 'dataset.pt'))
            dataset_converted = {}
            for key in dataset.keys():
                dataset_converted[key] = [Data(x=torch.ones(g.num_nodes, 1), y=g.gt_induced_le5[:, 1:], edge_index=g.edge_index).to_dict() for g in dataset[key]]
            torch.save(dataset_converted, os.path.join(output_folder, 'dataset_compatible.pt'))

def plot_sampled(plot_config):
    # TODO: parameterize more
    sample_path = plot_config['sample_path']
    pattern_index = [3, 6, 7, 14, 13, 16, 15, 17, 18, 31, 30, 29, 35, 36, 34, 38, 37, 40, 42, 41, 44, 43, 46, 45, 47, 48, 49, 50, 51, 52]
    # for dataset_name in tqdm([entry.name for entry in os.scandir(sample_path) if entry.is_dir()], desc='plotting sampled datasets'):
    for dataset_name in ['Set_1', 'Set_2', 'Set_3', 'Set_4', 'Set_5']:
        dataset = torch.load(os.path.join(sample_path, dataset_name, 'dataset.pt'))
        fig, axes = plt.subplots(5, 6, figsize=(30, 24))
        axes = axes.flatten()
        plot_objects = []
        splits = ['train', 'val', 'test']
        for i in range(30):
            gts = []
            for split in splits:
                gt_node = list(itertools.chain(*[g.gt_induced_le5[:, i].tolist() for g in dataset[split]]))
                gts.append(gt_node)
            max_value = max([max(gt) for gt in gts])
            num_bins = 10
            ceil_max = math.ceil(max_value/num_bins) * num_bins
            bins = list(range(0, ceil_max + 1, ceil_max//num_bins))
            axes[i].set_xlim(0, ceil_max)
            axes[i].set_yscale('log')
            for split, gt_node in zip(splits, gts):
                n, _, patches = axes[i].hist(gt_node, bins=bins, alpha=0.4)
                if i == 0:
                    plot_objects.append(patches[0])
                if split != 'val':
                    for rect, height in zip(patches, n):
                        axes[i].text(rect.get_x() + rect.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=6)
            ax_pattern = axes[i].inset_axes([0.7, 0.7, 0.3, 0.3])
            pattern = nx.graph_atlas(pattern_index[i])
            nx.draw(pattern, nx.spring_layout(pattern, seed=0), ax=ax_pattern, node_size=20, with_labels=False)
            ax_pattern.axis('off')
        
        fig.suptitle(f'Ground Truth Distribution: {dataset_name}', fontsize=20, fontweight='bold', y=0.02)
        fig.legend(plot_objects, splits, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=18)
        fig.text(0.959, 0.03, 'Local Counts', ha='center', va='center', fontsize=14, fontweight='bold')
        fig.text(0.023, 0.955, '#Nodes', ha='center', va='center', fontsize=14, fontweight='bold')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.05)

        output_path = os.path.join(plot_config['root_folder'], 'sample')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{dataset_name}.pdf'), format='pdf')

def run_experiments(algorithms):
    for alg in algorithms:
        if alg.get('skip', True) == False:
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

def plot_results(plot_config):
    prediction_path = plot_config['prediction_path']
    num_sets = len([entry.name for entry in os.scandir(prediction_path) if entry.is_dir()])

    # 1. runtime comparison
    fig, axes = plt.subplots(1, num_sets, figsize=(6*num_sets, 6))
    for index, dataset_name in tqdm(enumerate([f'Set_{i+1}' for i in range(num_sets)]), desc='plotting runtime', total=num_sets):
        runtime = {}
        for alg in ['ESCAPE', 'EVOKE', 'MOTIVO']:
            with open(os.path.join(prediction_path, dataset_name, alg, 'time.txt'), 'r') as file:
                runtime[alg] = round(float(file.readline().strip()), 3)

        with open(os.path.join(prediction_path, dataset_name, 'DeSCo', 'pretrained', 'time.txt'), 'r') as file:
            runtime['DeSCo-P'] = round(float(file.readline().strip().split()[0]), 3)
        with open(os.path.join(prediction_path, dataset_name, 'DeSCo', 'finetuned', 'time.txt'), 'r') as file:
            runtime['DeSCo-F'] = round(float(file.readline().strip().split()[0]), 3)

        algs = list(runtime.keys())
        values = list(runtime.values())
        ax = axes[index] if num_sets > 1 else axes
        bars = ax.bar(algs, values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'], alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        ax.set_title(dataset_name, fontsize=16, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xticklabels(algs, ha='center', fontsize=12)
    
    fig.text(0.0, 0.5, 'Runtime (seconds)', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle('Runtime Comparison', fontsize=20, fontweight='bold', y=0.15)
    plt.savefig(os.path.join(plot_config['root_folder'], 'prediction', 'runtime.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)

    # 2. accuracy distribution(MAE, Q-ERROE) for all 29 patterns
    num_nodes = torch.tensor([3] * 2 + [4] * 6 + [5] * 21)
    pattern_index = [6, 7, 14, 13, 16, 15, 17, 18, 31, 30, 29, 35, 36, 34, 38, 37, 40, 42, 41, 44, 43, 46, 45, 47, 48, 49, 50, 51, 52]
    color_palette = {'MOTIVO': '#FF9999', 'DeSCo-P': '#66B2FF', 'DeSCo-F': '#99FF99'}
    for index, dataset_name in tqdm(enumerate([f'Set_{i+1}' for i in range(num_sets)]), desc='plotting accuracy distribution', total=num_sets):
        fig, axes = plt.subplots(5, 6, figsize=(33, 25))
        axes = axes.flatten()
        results = {}

        dataset = torch.load(os.path.join('input', dataset_name, 'dataset.pt'))['test']
        gt = torch.cat([torch.sum(g.gt_induced_le5[:, 1:], 0).reshape(1, -1) for g in dataset])
        gt = torch.abs(gt) // num_nodes
        
        # read in MOTIVO results
        motivo = pd.read_csv(os.path.join(prediction_path, dataset_name, 'MOTIVO', 'prediction.csv'), index_col=0)
        motivo = torch.abs(torch.tensor(motivo.values))
        motivo_qerror = (gt + 1) / (motivo + 1)
        motivo_qerror = torch.where(motivo_qerror > 1, motivo_qerror, 1/motivo_qerror)
        motivo_mae = torch.abs(gt - motivo)
        results['MOTIVO'] = {'qerror': motivo_qerror, 'mae': motivo_mae}

        # read in DeSCo results
        desco_p = pd.read_csv(os.path.join(prediction_path, dataset_name, 'DeSCo', 'pretrained', 'prediction.csv'), index_col=0)
        desco_p = torch.abs(torch.tensor(desco_p.values))
        desco_p_qerror = (gt + 1) / (desco_p + 1)
        desco_p_qerror = torch.where(desco_p_qerror > 1, desco_p_qerror, 1/desco_p_qerror)
        desco_p_mae = torch.abs(gt - desco_p)
        results['DeSCo-P'] = {'qerror': desco_p_qerror, 'mae': desco_p_mae}
        desco_f = pd.read_csv(os.path.join(prediction_path, dataset_name, 'DeSCo', 'finetuned', 'prediction.csv'), index_col=0)
        desco_f = torch.abs(torch.tensor(desco_f.values))
        desco_f_qerror = (gt + 1) / (desco_f + 1)
        desco_f_qerror = torch.where(desco_f_qerror > 1, desco_f_qerror, 1/desco_f_qerror)
        desco_f_mae = torch.abs(gt - desco_f)
        results['DeSCo-F'] = {'qerror': desco_f_qerror, 'mae': desco_f_mae}

        num_algs = len(results)
        # Create empty lists to store legend handles and labels
        legend_handles = []
        legend_labels = []

        # draw boxplot with qerror and lineplot with mae, for each pattern
        for i in range(29):
            ax = axes[i]
            qerror_data = []
            mae_data = []
            gt_data = []
            for method, data in results.items():
                qerror_data.append(data['qerror'][:, i].numpy())
                mae_data.append(np.mean(data['mae'][:, i].numpy()))
                gt_data.append(np.mean(gt[:, i].numpy()))
            
            # Plot boxplot for Q-Error
            bp = ax.boxplot(qerror_data, positions=range(num_algs), widths=0.5, patch_artist=True)
            for patch, color in zip(bp['boxes'], color_palette.values()):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            # Add mean markers to the boxplot
            means = [np.mean(qerror) for qerror in qerror_data]
            mean_scatter = ax.scatter(range(num_algs), means, color='red', marker='D', s=50, zorder=3)
            if i == 0:
                legend_handles.append(mean_scatter)
                legend_labels.append('Mean Q-error')
            
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Plot lineplot for MAE and Mean Ground Truth
            ax2 = ax.twinx()
            ax2.set_yscale('log')
            mae_line = ax2.plot(range(num_algs), mae_data, color='#FFCC99', marker='x', linewidth=2, markersize=8, alpha=0.7)[0]
            gt_line = ax2.plot(range(num_algs), gt_data, color='#66CC66', marker='o', linewidth=2, markersize=6, alpha=0.7)[0]
            if i == 0:
                legend_handles.extend([mae_line, gt_line])
                legend_labels.extend(['MAE', 'Avg. Ground Truth'])
            
            # Annotations for MAE and Ground Truth
            for x, y in zip(range(num_algs), mae_data):
                ax2.annotate(f'{y:.1f}', (x, y), xytext=(0, -5), textcoords='offset points', ha='center', va='bottom', color='k', fontweight='bold')
            ax2.annotate(f'{gt_data[0]:.1f}', (0, gt_data[0]), xytext=(0, -5), textcoords='offset points', ha='center', va='top', color='k', fontweight='bold')
            
            ax_pattern = axes[i].inset_axes([0.7, 0.7, 0.3, 0.3])
            pattern = nx.graph_atlas(pattern_index[i])
            nx.draw(pattern, nx.spring_layout(pattern, seed=0), ax=ax_pattern, node_size=20, with_labels=False)
            ax_pattern.axis('off')

            ax.set_xlabel('')
            ax.set_title(f'Pattern {i+1}')
            ax.set_xticks(range(num_algs))
            ax.set_xticklabels(results.keys())
            ax2.yaxis.set_visible(False)
            
        # Add legend to the 30th subplot
        legend_ax = axes[29]
        legend_ax.axis('off')
        legend_ax.legend(legend_handles, legend_labels, loc='center', fontsize='large')

        plt.savefig(os.path.join('plots', 'prediction', f'{dataset_name}.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)


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
        plot_sampled(config['plot_config'])
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
        plot_results(config['plot_config'])
        logger.critical("Plotting results completed.")
    else:
        logger.critical("Plotting results skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running experiments")
    parser.add_argument("--config", default="configs/config.json", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)