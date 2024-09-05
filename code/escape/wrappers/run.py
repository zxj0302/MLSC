import time
import os
from tqdm import tqdm
import numpy as np
import math
import sys
from multiprocessing import Pool, cpu_count

matrices = dict()
matrices[3] = np.matrix('1 1 1 1 ;'
                        '0 1 2 3 ;'
                        '0 0 1 3 ;'
                        '0 0 0 1 ')

matrices[4] = np.matrix('1 1 1 1 1 1 1 1 1 1 1 ;'
                        '0 1 2 2 3 3 3 4 4 5 6 ;'
                        '0 0 1 0 0 0 1 1 2 2 3 ;'
                        '0 0 0 1 3 3 2 5 4 8 12 ;'
                        '0 0 0 0 1 0 0 1 0 2 4 ;'
                        '0 0 0 0 0 1 0 1 0 2 4 ;'
                        '0 0 0 0 0 0 1 2 4 6 12 ;'
                        '0 0 0 0 0 0 0 1 0 4 12 ;'
                        '0 0 0 0 0 0 0 0 1 1 3 ;'
                        '0 0 0 0 0 0 0 0 0 1 6 ;'
                        '0 0 0 0 0 0 0 0 0 0 1')

matrices[5] = np.matrix('1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ;'
                        '0 1 2 2 3 3 3 4 4 5 6 3 4 4 4 4 5 5 5 5 5 6 6 6 6 6 7 7 7 7 8 8 9 10 ;'
                        '0 0 1 0 0 0 1 1 2 2 3 2 3 0 2 3 2 4 3 4 5 5 5 4 6 6 6 6 7 8 10 9 12 15 ;'
                        '0 0 0 1 3 3 2 5 4 8 12 1 3 6 4 3 8 6 7 6 5 10 10 11 9 9 15 15 14 13 18 19 24 30 ;'
                        '0 0 0 0 1 0 0 1 0 2 4 0 1 0 0 0 1 1 1 0 0 2 2 2 1 0 3 4 3 2 4 5 7 10 ;'
                        '0 0 0 0 0 1 0 1 0 2 4 0 0 4 1 0 4 1 2 1 0 4 3 5 2 2 8 7 6 4 8 10 14 20 ;'
                        '0 0 0 0 0 0 1 2 4 6 12 0 0 0 2 2 4 4 5 6 5 8 10 10 10 12 18 18 17 18 28 28 42 60 ;'
                        '0 0 0 0 0 0 0 1 0 4 12 0 0 0 0 0 2 1 2 0 0 4 5 6 2 0 12 15 10 6 16 22 36 60 ;'
                        '0 0 0 0 0 0 0 0 1 1 3 0 0 0 0 0 0 0 0 1 0 0 1 1 1 3 3 3 2 3 5 5 9 15 ;'
                        '0 0 0 0 0 0 0 0 0 1 6 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 3 6 2 1 4 8 15 30 ;'
                        '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 2 5 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 1 3 0 1 2 1 4 2 3 5 6 5 3 7 6 6 6 9 11 16 13 21 30 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 2 1 0 1 0 0 1 2 2 4 3 6 10 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 2 1 1 0 1 2 3 5 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 1 2 2 0 4 4 5 4 6 12 9 10 10 20 20 36 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 1 2 5 4 4 2 7 6 6 6 10 14 24 18 36 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 2 0 2 0 0 6 3 3 0 4 8 15 30 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 4 2 0 2 0 0 3 6 6 16 12 30 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 2 1 0 6 6 5 4 12 14 30 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 2 6 6 3 4 8 16 12 30 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 2 4 2 6 12 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 2 2 6 15 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 3 2 2 8 8 24 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 6 3 2 0 4 10 24 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 4 12 6 24 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 2 1 4 10 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 3 10 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 6 20 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4 4 18 60 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 4 1 9 30 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 3 15 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 6 30 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 10 ;'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1'
                        )

#compute the inverse of the matrix
matrices[3] = np.linalg.inv(matrices[3])
matrices[4] = np.linalg.inv(matrices[4])
matrices[5] = np.linalg.inv(matrices[5])

dataset_name = sys.argv[1]
output_folder = sys.argv[2]

def process_graph(input):
    output = f'output/{dataset_name}/'+ input.replace('.edges', '.txt')
    start = time.time()
    os.system(f'../exe/count_five data/{dataset_name}/{input} {output}> /dev/null')
    time_noninduced = time.time() - start

    with open(output, 'r') as f_out:
        lines = f_out.readlines()

    noninduced = {3: [], 4: [], 5: []}
    n, m = 0, 0

    for num_lines, line in enumerate(lines, 1):
        current = float(line.strip())
        if num_lines == 1:
            n = current
        elif num_lines == 2:
            m = current
        elif 3 <= num_lines <= 6:
            noninduced[3].append(current)
        elif 7 <= num_lines <= 17:
            noninduced[4].append(current)
        elif 18 <= num_lines <= 51:
            noninduced[5].append(current)

    for i in range(3, 6):
        if len(noninduced[i]) > 0:
            noninduced[i] = np.array(noninduced[i])

    induced = {3: [], 4: [], 5: []}

    start = time.time()
    for i in range(3, 6):
        if len(noninduced[i]) > 0:
            induced[i] = matrices[i].dot(noninduced[i].T).T
    time_induced = time.time() - start

    return input.replace('.edges', ''), time_noninduced, time_induced

if __name__ == '__main__':
    graph_num = len(os.listdir('data/' + dataset_name))
    graphs = [str(i) + '.edges' for i in range(graph_num)]

    if not os.path.exists('output/' + dataset_name):
        os.makedirs('output/' + dataset_name)

    # Determine the number of processes to use (you can adjust this)
    num_processes = min(8, len(graphs))

    start_whole = time.time()

    # Create a pool of worker processes
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_graph, graphs), total=len(graphs), desc=f'{dataset_name}'))

    time_taken = time.time() - start_whole

    #reorder the results by the graph number
    results = sorted(results, key=lambda x: int(x[0]))
    # print(sum([x[1] for x in results]), sum([x[2] for x in results]))

    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder + '/time.txt', 'w') as f:
        f.write(str(time_taken) + '\n')
        for graph, time_noninduced, time_induced in results:
            f.write(f'{graph} {time_noninduced} {time_induced}\n')


# import time
# import os
# from tqdm import tqdm
# import numpy as np
# import math
# import sys

# matrices = dict()
# matrices[3] = np.matrix('1 1 1 1 ;'
#                         '0 1 2 3 ;'
#                         '0 0 1 3 ;'
#                         '0 0 0 1 ')

# matrices[4] = np.matrix('1 1 1 1 1 1 1 1 1 1 1 ;'
#                         '0 1 2 2 3 3 3 4 4 5 6 ;'
#                         '0 0 1 0 0 0 1 1 2 2 3 ;'
#                         '0 0 0 1 3 3 2 5 4 8 12 ;'
#                         '0 0 0 0 1 0 0 1 0 2 4 ;'
#                         '0 0 0 0 0 1 0 1 0 2 4 ;'
#                         '0 0 0 0 0 0 1 2 4 6 12 ;'
#                         '0 0 0 0 0 0 0 1 0 4 12 ;'
#                         '0 0 0 0 0 0 0 0 1 1 3 ;'
#                         '0 0 0 0 0 0 0 0 0 1 6 ;'
#                         '0 0 0 0 0 0 0 0 0 0 1')

# matrices[5] = np.matrix('1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ;'
#                         '0 1 2 2 3 3 3 4 4 5 6 3 4 4 4 4 5 5 5 5 5 6 6 6 6 6 7 7 7 7 8 8 9 10 ;'
#                         '0 0 1 0 0 0 1 1 2 2 3 2 3 0 2 3 2 4 3 4 5 5 5 4 6 6 6 6 7 8 10 9 12 15 ;'
#                         '0 0 0 1 3 3 2 5 4 8 12 1 3 6 4 3 8 6 7 6 5 10 10 11 9 9 15 15 14 13 18 19 24 30 ;'
#                         '0 0 0 0 1 0 0 1 0 2 4 0 1 0 0 0 1 1 1 0 0 2 2 2 1 0 3 4 3 2 4 5 7 10 ;'
#                         '0 0 0 0 0 1 0 1 0 2 4 0 0 4 1 0 4 1 2 1 0 4 3 5 2 2 8 7 6 4 8 10 14 20 ;'
#                         '0 0 0 0 0 0 1 2 4 6 12 0 0 0 2 2 4 4 5 6 5 8 10 10 10 12 18 18 17 18 28 28 42 60 ;'
#                         '0 0 0 0 0 0 0 1 0 4 12 0 0 0 0 0 2 1 2 0 0 4 5 6 2 0 12 15 10 6 16 22 36 60 ;'
#                         '0 0 0 0 0 0 0 0 1 1 3 0 0 0 0 0 0 0 0 1 0 0 1 1 1 3 3 3 2 3 5 5 9 15 ;'
#                         '0 0 0 0 0 0 0 0 0 1 6 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 3 6 2 1 4 8 15 30 ;'
#                         '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 2 5 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 1 3 0 1 2 1 4 2 3 5 6 5 3 7 6 6 6 9 11 16 13 21 30 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 2 1 0 1 0 0 1 2 2 4 3 6 10 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 2 1 1 0 1 2 3 5 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 1 2 2 0 4 4 5 4 6 12 9 10 10 20 20 36 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 1 2 5 4 4 2 7 6 6 6 10 14 24 18 36 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 2 0 2 0 0 6 3 3 0 4 8 15 30 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 4 2 0 2 0 0 3 6 6 16 12 30 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 2 1 0 6 6 5 4 12 14 30 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 2 6 6 3 4 8 16 12 30 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 2 4 2 6 12 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 2 2 6 15 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 3 2 2 8 8 24 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 6 3 2 0 4 10 24 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 4 12 6 24 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 2 1 4 10 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 3 10 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 2 6 20 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4 4 18 60 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 4 1 9 30 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 3 15 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 6 30 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 10 ;'
#                         '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1'
#                         )

# #compute the inverse of the matrix
# matrices[3] = np.linalg.inv(matrices[3])
# matrices[4] = np.linalg.inv(matrices[4])
# matrices[5] = np.linalg.inv(matrices[5])

# dataset_name = sys.argv[1]
# output_folder = sys.argv[2]
# # for d in dataset_list:
#     # dir_path = f'data/{d}'
# graph_num = len(os.listdir('data/'+dataset_name))
# start_whole = time.time()
# # for all the files in the directory
# graphs = [str(i)+'.edges' for i in range(graph_num)]
# for file in graphs:
#     # execute the escape algorithm
#     # start_exec = time.time()
#     os.system('../exe/count_five data/'+dataset_name+'/'+file+' > /dev/null')
#     # print('non-induced: ' + str(time.time()-start_exec))

#     f_out = open('out.txt','r')

#     noninduced = dict() # storing all noninduced in a dictionary of lists
#     noninduced[3] = []  # list for each possible size
#     noninduced[4] = []
#     noninduced[5] = []

#     num_lines = 1
#     for line in f_out.readlines():
#         current = float(line.strip())
#         if num_lines == 1:
#             n = current
#         elif num_lines == 2:
#             m = current
#         elif num_lines >= 3 and num_lines <= 6:
#             noninduced[3].append(current)
#         elif num_lines >= 7 and num_lines <= 17:
#             noninduced[4].append(current)
#         elif num_lines >= 18 and num_lines <= 51:
#             noninduced[5].append(current)
#         num_lines = num_lines + 1

#     f_out.close()

#     # converto numpy arrays
#     for i in range(3,6):
#         if len(noninduced[i]) > 0:
#             noninduced[i] = np.array(noninduced[i])

#     induced = dict()
#     induced[3] = []
#     induced[4] = []
#     induced[5] = []

#     # start_induced = time.time()
#     for i in range(3,6):
#         if len(noninduced[i]) > 0:
#             induced[i] = matrices[i].dot(noninduced[i].T).T
#             # induced[i] = np.linalg.solve(matrices[i],noninduced[i])  #inverting matrices[i] to convert non-induced to induced noninduced
#     print(induced[3])
#     print(induced[4])
#     print(induced[5])
#     # print('induced: ' + str(time.time()-start_induced))


# time_taken = time.time() - start_whole
# print('time taken: '+str(time_taken)+' seconds')