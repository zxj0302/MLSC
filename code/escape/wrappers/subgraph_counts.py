# This is the main wrapper function for getting subgraph counts. It simply calls
# the appropriate executables, which output the number of non-induced patterns. This code
# applies appropriate linear transformations to get all induced counts.
# 
# The function should be called as follows:
# 
# python3 subgraph_counts.py <PATH TO INPUT> <DESIRED PATTERN SIZE> <OPTIONAL FLAGS>
# 
# DESIRED PATTERN SIZE: Either 3, 4, or 5
# OPTIONAL FLAGS:
#         -i: output counts as integers. Useful for small graphs, or for debugging
# 
# 
# Output format:
# 
# Each line contains the name of a pattern, the number of non-induced occurrences, the number
# of induced occurrences, and the ratio between these.
# 

import os
import numpy as np
import sys
import math

matrices = dict() # conversion matrices: converting induced to non-induced

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

if len(sys.argv) < 3:
    print( 'Format: python3 subgraph_counts.py <PATH FOR INPUT> <DESIRED PATTERN SIZE> <OPTIONS>\n DESIRED PATTERN SIZE: 3, 4, or 5\n OPTIONS:\n \t -i: for integer outputs, useful in looking at small graphs' )
    sys.exit()

integral = False
if '-i' in sys.argv:
    integral = True

pattern = int(sys.argv[2])  # size of desired patterns

if pattern == 3:   # calling appropriate executable, output is in out.txt
    os.system('../exe/count_three '+sys.argv[1])
elif pattern == 4:
    os.system('../exe/count_four '+sys.argv[1])
elif pattern == 5:
    os.system('../exe/count_five '+sys.argv[1])
else:
    print('Incorrect format: Desired pattern size must be 3, 4, or 5')
    sys.exit()

f_out = open('out.txt','r')

noninduced = dict() # storing all noninduced in a dictionary of lists

noninduced[3] = list()  # list for each possible size
noninduced[4] = list()
noninduced[5] = list()

num_lines = 1
for line in f_out.readlines():
    current = float(line.strip())
    if num_lines == 1:
        n = current
    if num_lines == 2:
        m = current
    if num_lines >= 3 and num_lines <= 6:
        noninduced[3].append(current)
    if num_lines >= 7 and num_lines <= 17:
        noninduced[4].append(current)
    if num_lines >= 18 and num_lines <= 51:
        noninduced[5].append(current)
    num_lines = num_lines + 1

f_out.close()

# converto numpy arrays
for i in range(3,6):
    if len(noninduced[i]) > 0:
        noninduced[i] = np.array(noninduced[i])

induced = dict()
induced[3] = list()
induced[4] = list()
induced[5] = list()


for i in range(3,6):
    if len(noninduced[i]) > 0:
        induced[i] = np.linalg.solve(matrices[i],noninduced[i])  #inverting matrices[i] to convert non-induced to induced noninduced

print('==============')
print('Basic size')
print('==============')

print('Vertices\t',math.floor(n))
print('Edges\t\t',math.floor(m))

names = dict()

names[3] = ['Ind set\t\t', 'Only edge\t', 'Wedge\t\t', 'Triangle\t']
names[4] = ['Ind set\t\t', 'Only edge\t', 'Matching\t', 'Only wedge\t', 'Only triangle\t', '3-star\t\t', '3-path\t\t', 'Tailed triangle\t', '4-cycle\t\t', 'Diamond\t\t', '4-clique\t'] 
names[5] = ['Ind set\t\t', 'Only edge\t', 'Matching\t', 'Only wedge\t', 'Only triangle\t', 'Only 3-star\t', 'Only 3-path\t', 'Only Tailed tri\t', 'Only 4-cycle\t', 'Only Diamond\t', 'Only 4-clique\t', 'Wedge+edge\t', 'Triangle+edge\t', '4-star\t\t', 'Prong\t\t', '4-path\t\t', 'Forktailed-tri\t','Lontailed-tri\t','Doubletailed-tri','Tailed-4-cycle\t','5-cycle\t\t','Hourglass\t','Cobra\t\t','Stingray\t','Hatted-4-cycle\t','3-wedge-col\t','3-tri-collision\t','Tailed-4-clique\t','Triangle-strip\t','Diamond-wed-col\t','4-wheel\t\t','Hatted-4-clique\t','Almost-5-clique\t','5-clique\t'] 



for i in range(3,6):
    if len(induced[i]) == 0:
        break
    print('==============')
    print(str(i)+' vertex patterns')
    print('==============')
    for j in range(0,len(induced[i])):
        if noninduced[i][j] == 0:
            if integral:
                print(names[i][j],math.floor(noninduced[i][j]),math.floor(induced[i][j]),'-')
            else:
                print(names[i][j],'%.2e'%noninduced[i][j],'%.2e'%induced[i][j],'-')
        else:
            if integral:
                print(names[i][j],math.floor(noninduced[i][j]),math.floor(induced[i][j]),'%.2f'%(induced[i][j]/noninduced[i][j]))
            else:
                print(names[i][j],'%.2e'%noninduced[i][j],'%.2e'%induced[i][j],'%.2f'%(induced[i][j]/noninduced[i][j]))
                 










