import os
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import sys
from multiprocessing import Pool, cpu_count

# get the first argument which is the edgelist type, graph or desco-egonetwork
dataset_name = sys.argv[1]
edgelist_path = f'data/{dataset_name}'
num_file = len(os.listdir(edgelist_path))
output_folder = sys.argv[2]
num_cpus = int(sys.argv[3]) if len(sys.argv) > 3 else min(8, cpu_count())
os.makedirs(f'input_bin/{dataset_name}', exist_ok=True)
os.makedirs(f'output/{dataset_name}', exist_ok=True)

def process_file(i):
    # start = time.time()
    # os.system(f'../build/bin/motivo-graph -f LOE --input {edgelist_path}/{i}.edges --output input_bin/{dataset_name}/input_{i} > /dev/null')
    # time_conversion = time.time() - start

    # start = time.time()
    # os.system(f'../scripts/motivo.sh -g input_bin/{dataset_name}/input_{i} -k 8 -o output/{dataset_name}/{i} --build > /dev/null')
    # time_build = time.time() - start
    
    # read the number of nodes and set sample number
    num_samples = max(int(open(f'{edgelist_path}/{i}.edges', 'r').readline().split()[0]) * 10, 1000000)

    # start = time.time()
    # os.system(f'../scripts/motivo.sh -g input_bin/{dataset_name}/input_{i} -k 3 -o output/{dataset_name}/{i} --sample -s {num_samples} -a > /dev/null')
    # time_3 = time.time() - start
    # if os.path.exists(f'output/{dataset_name}/{i}.csv'):
    #     os.system(f'mv output/{dataset_name}/{i}.csv output/{dataset_name}/{i}_3.csv')

    # start = time.time()
    # os.system(f'../scripts/motivo.sh -g input_bin/{dataset_name}/input_{i} -k 4 -o output/{dataset_name}/{i} --sample -s {num_samples} -a > /dev/null')
    # time_4 = time.time() - start
    # if os.path.exists(f'output/{dataset_name}/{i}.csv'):
    #     os.system(f'mv output/{dataset_name}/{i}.csv output/{dataset_name}/{i}_4.csv')

    # start = time.time()
    # os.system(f'../scripts/motivo.sh -g input_bin/{dataset_name}/input_{i} -k 5 -o output/{dataset_name}/{i} --sample -s {num_samples} -a > /dev/null')
    # time_5 = time.time() - start
    # if os.path.exists(f'output/{dataset_name}/{i}.csv'):
    #     os.system(f'mv output/{dataset_name}/{i}.csv output/{dataset_name}/{i}_5.csv')

    os.system(f'../scripts/motivo.sh -g input_bin/{dataset_name}/input_{i} -k 6 -o output/{dataset_name}/{i} --sample -s {num_samples} -a > /dev/null')
    if os.path.exists(f'output/{dataset_name}/{i}.csv'):
        os.system(f'mv output/{dataset_name}/{i}.csv output/{dataset_name}/{i}_6.csv')

    os.system(f'../scripts/motivo.sh -g input_bin/{dataset_name}/input_{i} -k 7 -o output/{dataset_name}/{i} --sample -s {num_samples} -a > /dev/null')
    if os.path.exists(f'output/{dataset_name}/{i}.csv'):
        os.system(f'mv output/{dataset_name}/{i}.csv output/{dataset_name}/{i}_7.csv')

    os.system(f'../scripts/motivo.sh -g input_bin/{dataset_name}/input_{i} -k 8 -o output/{dataset_name}/{i} --sample -s {num_samples} -a > /dev/null')
    if os.path.exists(f'output/{dataset_name}/{i}.csv'):
        os.system(f'mv output/{dataset_name}/{i}.csv output/{dataset_name}/{i}_8.csv')

    return i, 0, 0, 0, 0, 0  # time_conversion, time_build, time_3, time_4, time_5


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
        for r in results:
            f.write(f'{r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]}\n')

    patterns = ['G', 'O', 'EM', 'BM', 'MM', 'DM', 'HM', 'PM', 'BFI', 'AHE', 'ADM', 'BFM', 'IFM', 'AHM', 'MIM', 'API', 'APM', 'IHM', 'COM', 'BPI', 'JFM', 'BPM', 'CPM', 'EPM', 'MNM', 'DPM', 'MPM', 'HPM', 'PPM']
    header = ['index'] + patterns
    # make a csv with patterns as header and the number of occurrences as the value
    predictions = pd.DataFrame(columns=header, index=range(num_file))
    predictions['index'] = range(num_file)

    for i in range(num_file):
        for j in range(3, 6):
            file_path = f'output/{dataset_name}/{i}_{j}.csv'
            try:
                data = pd.read_csv(file_path, usecols=[0, 1], header=0)
                for row in data.iterrows():
                    predictions.loc[i, row[1][0]] = row[1][1]
            except:
                pass
    
    # fill the NaN values with 0
    predictions = predictions.fillna(0)
    predictions.to_csv(output_folder + '/prediction.csv', index=False)