import os
import time
import numpy as np
from tqdm import tqdm
import torch
import sys
from multiprocessing import Pool
import subprocess
import re

matrices = dict()
matrices[3] = np.matrix('1 0 2 ;'
                        '0 1 1 ;'
                        '0 0 1 ')
# changed matrices[1][9] from 2 to 4, because their code is wrong
matrices[4] = np.matrix('1 0 0 0 2 2 1 0 4 2 6 ;'
                        '0 1 0 0 2 0 1 2 2 4 6 ;'
                        '0 0 1 0 0 1 1 0 2 1 3 ;'
                        '0 0 0 1 0 0 0 1 0 1 1 ;'
                        '0 0 0 0 1 0 0 0 1 1 3 ;'
                        '0 0 0 0 0 1 0 0 2 0 3 ;'
                        '0 0 0 0 0 0 1 0 2 2 6 ;'
                        '0 0 0 0 0 0 0 1 0 2 3 ;'
                        '0 0 0 0 0 0 0 0 1 0 3 ;'
                        '0 0 0 0 0 0 0 0 0 1 3 ;'
                        '0 0 0 0 0 0 0 0 0 0 1')
matrices[5] = np.matrix(
    '1 0 0 0 0 0 0 0 0 1 0 0 2 0 1 0 0 0 0 2 2 0 1 0 2 1 0 0 2 0 4 2 0 1 4 0 3 4 2 4 0 6 2 0 6 3 2 8 4 6 12 8 4 10 8 '
    '18 12 24 ;'
    '0 1 0 0 0 0 0 0 0 0 0 1 0 2 1 0 0 0 0 2 0 2 0 2 0 0 2 2 2 0 0 2 4 1 0 6 3 2 3 0 6 0 2 6 3 5 4 4 8 4 4 6 10 10 8 '
    '12 16 24 ;'
    '0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 2 0 0 0 1 0 0 1 0 0 1 0 0 0 4 0 0 0 2 2 0 1 1 2 2 0 0 2 0 1 2 4 2 2 4 2 4 4 4 8 6 '
    '8 12 ;'
    '0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 2 0 0 1 1 2 0 1 0 2 2 0 0 0 3 2 0 0 2 3 3 2 0 3 2 0 4 3 0 6 4 3 5 0 9 '
    '6 12 ;'
    '0 0 0 0 1 0 0 0 0 1 2 0 0 0 1 0 2 0 0 0 2 0 1 0 4 3 0 0 2 0 2 2 0 2 4 0 1 4 1 8 0 6 4 0 6 3 2 4 2 6 12 10 4 8 8 '
    '18 12 24 ;'
    '0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 1 0 0 1 2 1 1 0 0 0 2 1 2 0 1 0 1 2 3 0 2 3 1 3 2 2 2 2 2 4 5 4 4 6 '
    '8 12 ;'
    '0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 2 0 0 0 0 2 0 0 1 4 0 4 0 0 2 1 0 3 0 0 2 0 6 0 1 6 0 2 6 0 3 2 0 2 8 3 8 3 '
    '10 12 ;'
    '0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 2 1 1 1 0 1 1 0 0 0 0 2 2 1 1 0 3 '
    '2 4 ;'
    '0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 '
    '1 1 ;'
    '0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 1 0 0 0 0 2 2 0 0 0 0 1 0 0 4 0 6 2 0 4 1 0 4 2 0 12 6 2 6 0 '
    '18 8 24 ;'
    '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 2 0 0 2 0 1 1 1 0 0 2 2 4 2 2 4 6 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 2 1 0 0 0 0 1 0 6 0 2 6 0 3 4 0 2 2 0 4 10 4 8 6 '
    '16 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 0 0 0 0 0 1 0 0 0 0 3 0 0 2 1 0 4 1 0 6 2 1 4 0 9 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 0 0 0 1 0 0 0 0 0 0 3 1 2 0 2 2 0 2 2 3 4 0 6 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 2 0 1 0 0 0 2 1 0 0 0 2 0 3 2 2 0 2 4 4 6 4 6 8 12 '
    '12 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 4 0 1 2 0 2 4 2 8 3 '
    '8 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 2 1 0 0 1 0 0 0 0 0 0 0 0 0 0 4 0 3 1 0 2 1 0 0 0 0 6 4 1 2 0 9 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 2 0 1 0 0 0 0 0 0 0 0 0 0 2 3 0 2 0 1 2 0 0 0 0 2 4 3 2 0 6 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 2 0 2 0 0 0 0 0 0 0 0 0 0 3 0 0 3 0 0 3 0 0 0 0 0 4 0 4 0 '
    '5 6 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 1 1 2 2 2 2 2 2 4 4 6 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 2 0 0 2 0 2 0 3 0 0 2 0 0 2 0 3 6 3 0 3 4 9 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 3 1 0 0 0 3 0 1 0 1 1 0 2 3 0 2 2 3 4 0 6 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 4 0 1 0 1 4 0 0 2 0 1 2 2 4 2 4 4 6 4 6 8 12 '
    '12 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 3 0 0 1 0 3 0 0 3 0 1 2 0 3 1 0 1 5 3 4 3 '
    '8 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 3 0 0 1 0 0 0 0 0 6 2 0 1 0 9 '
    '2 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 2 0 1 1 0 0 0 0 4 6 2 2 0 12 '
    '8 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 1 0 0 1 0 0 0 0 0 2 3 1 0 3 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 3 0 0 2 0 0 0 0 0 5 0 4 0 '
    '8 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 2 2 1 2 0 6 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 2 0 '
    '2 3 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 3 0 0 1 0 0 2 0 0 6 1 0 2 0 9 '
    '2 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 2 2 1 2 0 6 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 3 0 1 0 0 1 0 0 1 3 2 0 3 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 2 0 0 1 2 0 0 2 0 4 4 2 8 6 '
    '12 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 1 1 0 1 2 3 '
    '2 6 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 1 '
    '2 4 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 4 2 0 4 2 2 6 0 12 '
    '8 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 2 2 2 0 2 4 6 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 2 0 2 2 0 2 4 4 8 6 '
    '12 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 3 '
    '1 6 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 '
    '2 4 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 2 0 0 0 0 3 '
    '0 4 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 2 1 0 0 3 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 '
    '2 4 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 4 2 0 2 0 12 '
    '4 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 2 2 2 0 6 '
    '8 24 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 2 0 4 0 '
    '6 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 3 '
    '1 6 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 2 0 3 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 4 3 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 3 '
    '0 6 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 3 '
    '2 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 '
    '4 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 3 '
    '2 12 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 '
    '1 3 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 '
    '0 4 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '1 6 ;'
    '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 '
    '0 1')
matrices[3] = np.linalg.inv(matrices[3])
matrices[4] = np.linalg.inv(matrices[4])
matrices[5] = np.linalg.inv(matrices[5])


# get the first argument which is the edgelist type, graph or desco-egonetwork
dataset_name = sys.argv[1]
edgelist_path = f'data/{dataset_name}'
edgelist_processed_path = f'data/{dataset_name}_processed'
os.makedirs(edgelist_processed_path, exist_ok=True)
num_file = len(os.listdir(edgelist_path))
output_folder = sys.argv[2]
# os.makedirs('output_scope', exist_ok=True)
num_cpus = int(sys.argv[3]) if len(sys.argv) > 3 else 8
query = sys.argv[4] if len(sys.argv) > 4 else ""
results_path = sys.argv[5] if len(sys.argv) > 5 else "wrappers/output_scope"
os.makedirs(results_path, exist_ok=True)
use_5voc = int(sys.argv[6]) if len(sys.argv) > 6 else 0
if use_5voc:
    program = '5voc'
else:
    program = 'scope'

def process_file(i):
    # Part 1: Execute the orbit counting command
    result_path_i = f'{results_path}/{dataset_name}/out_{i}'
    os.makedirs(result_path_i, exist_ok=True)
    start = time.time()
    os.system(f'../build/executable/preprocess.out {edgelist_path}/{i}.edges {edgelist_processed_path}/{i}_reordered.edges {edgelist_processed_path}/{i}_reordered.bin {edgelist_processed_path}/{i}_node_map.txt > /dev/null')
    preprocess_time = time.time() - start
    # start = time.time()
    # os.system(f'../build/executable/scope.out -q {query} -d {edgelist_processed_path}/{i}_reordered.edges -t {edgelist_processed_path}/{i}_reordered.bin -r {result_path_i} -b -share> /dev/null')
    # time_orbit = time.time() - start
    result = subprocess.run(
        f'../build/executable/{program}.out -q {query} -d {edgelist_processed_path}/{i}_reordered.edges -t {edgelist_processed_path}/{i}_reordered.bin -b -share',
        shell=True,
        capture_output=True,
        text=True
    )
    output = result.stdout
    time_match = re.search(r'total time:\s*([\d.]+)', output)
    if time_match:
        time_orbit = float(time_match.group(1))
    else:
        # 如果没有找到 total time，可以作为备选方案记录错误或使用其他方法
        print("Warning: Could not extract time from scope.out output")
        time_orbit = None  # 或者你可以选择其他默认值

    # Part 2: Read and process the output
    # with open(result_path_i, 'r') as f_in:
        # non_induced = np.array([np.array([int(float(x)) for x in line.split()]) for line in f_in])

    # start = time.time()
    # # compute the induced orbit counts
    # non_induced_t = np.transpose(non_induced)
    # induced_3 = matrices[3] @ non_induced_t[1:4, :]
    # induced_4 = matrices[4] @ non_induced_t[4:15, :]
    # induced_5 = matrices[5] @ non_induced_t[15:, :]
    # induced = np.concatenate((non_induced_t[:1, :], induced_3, induced_4, induced_5), axis=0).T

    # # convert the orbit counts to pattern counts
    # noninduced_patterns = torch.tensor([[x[0], x[1]+x[2], x[3], x[4]+x[5], x[6]+x[7], x[8], x[9]+x[10]+x[11], x[12]+x[13],
    #                                      x[14], x[15]+x[16]+x[17], x[18]+x[19]+x[20]+x[21], x[22]+x[23], x[24]+x[25]+x[26],
    #                                      x[27]+x[28]+x[29]+x[30], x[31]+x[32]+x[33], x[34], x[35]+x[36]+x[37]+x[38],
    #                                      x[39]+x[40]+x[41]+x[42], x[43]+x[44], x[45]+x[46]+x[47]+x[48], x[49]+x[50],
    #                                      x[51]+x[52]+x[53], x[54]+x[55], x[56]+x[57]+x[58], x[59]+x[60]+x[61],
    #                                      x[62]+x[63]+x[64], x[65]+x[66]+x[67], x[68]+x[69], x[70]+x[71], x[72]]
    #                                     for x in non_induced.tolist()], dtype=torch.int64).T
    # induced_patterns = torch.tensor([[x[0], x[1]+x[2], x[3], x[4]+x[5], x[6]+x[7], x[8], x[9]+x[10]+x[11], x[12]+x[13],
    #                                   x[14], x[15]+x[16]+x[17], x[18]+x[19]+x[20]+x[21], x[22]+x[23], x[24]+x[25]+x[26],
    #                                   x[27]+x[28]+x[29]+x[30], x[31]+x[32]+x[33], x[34], x[35]+x[36]+x[37]+x[38],
    #                                   x[39]+x[40]+x[41]+x[42], x[43]+x[44], x[45]+x[46]+x[47]+x[48], x[49]+x[50],
    #                                   x[51]+x[52]+x[53], x[54]+x[55], x[56]+x[57]+x[58], x[59]+x[60]+x[61],
    #                                   x[62]+x[63]+x[64], x[65]+x[66]+x[67], x[68]+x[69], x[70]+x[71], x[72]]
    #                                  for x in induced.tolist()], dtype=torch.int64).T
    # time_conversion = time.time() - start

    return i, preprocess_time, time_orbit


if __name__ == '__main__':
    start_whole = time.time()
    with Pool(min(num_cpus, num_file)) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, range(num_file)), 
                            total=num_file, 
                            desc=dataset_name))
    time_taken = time.time() - start_whole

    results = sorted(results, key=lambda x: int(x[0]))

    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder + '/time.txt', 'w') as f:
        f.write(str(time_taken) + '\n')
        for graph, preprocess, time_noninduced in results:
            f.write(f'{graph} {preprocess} {time_noninduced}\n')
