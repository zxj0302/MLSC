import os
import time
import numpy as np
from tqdm import tqdm
import torch
import sys
from multiprocessing import Pool, cpu_count

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
num_file = len(os.listdir(edgelist_path))
output_folder = sys.argv[2]
os.makedirs('output_evoke', exist_ok=True)

def process_file(i):
    # Part 1: Execute the orbit counting command
    start = time.time()
    os.system(f'../exe/count_orbit_five {edgelist_path}/{i}.edges > /dev/null')
    time_orbit = time.time() - start

    # Part 2: Read and process the output
    with open(f'output_evoke/out_{i}', 'r') as f_in:
        non_induced = np.array([np.array([int(float(x)) for x in line.split()]) for line in f_in])

    start = time.time()
    # compute the induced orbit counts
    non_induced_t = np.transpose(non_induced)
    induced_3 = matrices[3] @ non_induced_t[1:4, :]
    induced_4 = matrices[4] @ non_induced_t[4:15, :]
    induced_5 = matrices[5] @ non_induced_t[15:, :]
    induced = np.concatenate((non_induced_t[:1, :], induced_3, induced_4, induced_5), axis=0).T

    # convert the orbit counts to pattern counts
    noninduced_patterns = torch.tensor([[x[0], x[1]+x[2], x[3], x[4]+x[5], x[6]+x[7], x[8], x[9]+x[10]+x[11], x[12]+x[13],
                                         x[14], x[15]+x[16]+x[17], x[18]+x[19]+x[20]+x[21], x[22]+x[23], x[24]+x[25]+x[26],
                                         x[27]+x[28]+x[29]+x[30], x[31]+x[32]+x[33], x[34], x[35]+x[36]+x[37]+x[38],
                                         x[39]+x[40]+x[41]+x[42], x[43]+x[44], x[45]+x[46]+x[47]+x[48], x[49]+x[50],
                                         x[51]+x[52]+x[53], x[54]+x[55], x[56]+x[57]+x[58], x[59]+x[60]+x[61],
                                         x[62]+x[63]+x[64], x[65]+x[66]+x[67], x[68]+x[69], x[70]+x[71], x[72]]
                                        for x in non_induced.tolist()], dtype=torch.int64).T
    induced_patterns = torch.tensor([[x[0], x[1]+x[2], x[3], x[4]+x[5], x[6]+x[7], x[8], x[9]+x[10]+x[11], x[12]+x[13],
                                      x[14], x[15]+x[16]+x[17], x[18]+x[19]+x[20]+x[21], x[22]+x[23], x[24]+x[25]+x[26],
                                      x[27]+x[28]+x[29]+x[30], x[31]+x[32]+x[33], x[34], x[35]+x[36]+x[37]+x[38],
                                      x[39]+x[40]+x[41]+x[42], x[43]+x[44], x[45]+x[46]+x[47]+x[48], x[49]+x[50],
                                      x[51]+x[52]+x[53], x[54]+x[55], x[56]+x[57]+x[58], x[59]+x[60]+x[61],
                                      x[62]+x[63]+x[64], x[65]+x[66]+x[67], x[68]+x[69], x[70]+x[71], x[72]]
                                     for x in induced.tolist()], dtype=torch.int64).T
    time_conversion = time.time() - start

    return i, time_orbit, time_conversion


if __name__ == '__main__':
    start_whole = time.time()
    with Pool(8) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, range(num_file)), 
                            total=num_file, 
                            desc=dataset_name))
    time_taken = time.time() - start_whole

    results = sorted(results, key=lambda x: int(x[0]))

    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder + '/time.txt', 'w') as f:
        f.write(str(time_taken) + '\n')
        for graph, time_noninduced, time_induced in results:
            f.write(f'{graph} {time_noninduced} {time_induced}\n')
    



# # run the program on each files in the edgelist_path folder
# num_file = len(os.listdir(edgelist_path))

# #Part1: multiprocessing for ground truth computation
# with Pool(8) as p:
#     res = list(tqdm(p.imap_unordered(os.system, [f'../exe/count_orbit_five {edgelist_path}/{i}.edges opnmp > '
#         f'/dev/null' for i in range(num_file)]), total=num_file, desc='Non-induced orbit counts'))

# #Part2: read in the non-induced counts, organize and store
# for i in tqdm(range(num_file), desc='Non-induced orbit -> (Non-)Induced pattern counts'):
#     # read in the output file to get the non-induced orbit counts
#     with open(f'output_evoke/out_{i}', 'r') as f_in:
#         non_induced = np.array([np.array([int(float(x)) for x in line.split()]) for line in f_in])

#     # compute the induced orbit counts
#     non_induced_t = np.transpose(non_induced)
#     induced_3 = np.linalg.solve(matrices[3], non_induced_t[1:4, :])
#     induced_4 = np.linalg.solve(matrices[4], non_induced_t[4:15, :])
#     induced_5 = np.linalg.solve(matrices[5], non_induced_t[15:, :])
#     induced = np.concatenate((non_induced_t[:1, :], induced_3, induced_4, induced_5), axis=0).T

#     # convert the orbit counts to pattern counts
#     noninduced_patterns = torch.tensor([[x[0], x[1]+x[2], x[3], x[4]+x[5], x[6]+x[7], x[8], x[9]+x[10]+x[11], x[12]+x[13],
#                                          x[14], x[15]+x[16]+x[17], x[18]+x[19]+x[20]+x[21], x[22]+x[23], x[24]+x[25]+x[26],
#                                          x[27]+x[28]+x[29]+x[30], x[31]+x[32]+x[33], x[34], x[35]+x[36]+x[37]+x[38],
#                                          x[39]+x[40]+x[41]+x[42], x[43]+x[44], x[45]+x[46]+x[47]+x[48], x[49]+x[50],
#                                          x[51]+x[52]+x[53], x[54]+x[55], x[56]+x[57]+x[58], x[59]+x[60]+x[61],
#                                          x[62]+x[63]+x[64], x[65]+x[66]+x[67], x[68]+x[69], x[70]+x[71], x[72]]
#                                         for x in non_induced.tolist()], dtype=torch.int64).T
#     induced_patterns = torch.tensor([[x[0], x[1]+x[2], x[3], x[4]+x[5], x[6]+x[7], x[8], x[9]+x[10]+x[11], x[12]+x[13],
#                                       x[14], x[15]+x[16]+x[17], x[18]+x[19]+x[20]+x[21], x[22]+x[23], x[24]+x[25]+x[26],
#                                       x[27]+x[28]+x[29]+x[30], x[31]+x[32]+x[33], x[34], x[35]+x[36]+x[37]+x[38],
#                                       x[39]+x[40]+x[41]+x[42], x[43]+x[44], x[45]+x[46]+x[47]+x[48], x[49]+x[50],
#                                       x[51]+x[52]+x[53], x[54]+x[55], x[56]+x[57]+x[58], x[59]+x[60]+x[61],
#                                       x[62]+x[63]+x[64], x[65]+x[66]+x[67], x[68]+x[69], x[70]+x[71], x[72]]
#                                      for x in induced.tolist()], dtype=torch.int64).T

