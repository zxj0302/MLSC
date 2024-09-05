# This is a script to plot various orbit count distributions associated with one (two) graph(s)
# USAGE:
#         python3 plot_dd_two_graphs_normalized.py <ORBIT NUMBER> <GRAPH ADDRESS 1> <GRAPH NAME 1> <GRAPH ADDRESS 2> <GRAPH NAME 2>
# 
# The script creates the following figures in the figures/ directory.
#     orbit_<ORBIT NUMBER>_count_dist_<GRAPH NAME 1>_<GRAPH NAME 2>.png: This has the plots of the normalized orbit count distribution

# Noujan Pashanasangi, June 2019
#
#

import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np

total_counts = 0

def plot_dist(input_file, line_color):
    f_in = open(input_file, 'r')

    counts = list()
    count_dist_dict = dict()
    max_count = 0
    total_vert = 0;

  
    for line in f_in.readlines():
            tokens = line.split(' ')
            if(len(tokens) < 70):
                print(tokens)
                print(len(tokens))
            index = int(float(orbit))
            counts.append(int(float(tokens[index])))

    total_vert = len(counts)

    for count in counts:
        if count not in count_dist_dict:
            count_dist_dict[count] = 0
        count_dist_dict[count] = count_dist_dict[count] + 1
        if count > max_count:
            max_count = count

    items = sorted(count_dist_dict.items())

    f_in.close()


    # fig = plt.figure (1)
    plot1, = plt.plot([ k for (k , v ) in items ] , [ v/total_vert for (k ,v ) in items ], linestyle=':', linewidth=1, color= line_color)
    return plot1














if len(sys.argv) < 3: # args not correct
    print('USAGE: python3 plot_dd.py  <ORBIT NUMBER> <FILE ADDRESS 1> <graph_1_name> <FILE ADDRESS 2> <graph_1_name>')
    sys.exit(-1)

orbit = sys.argv[1]
double = False

graph_1_adrs = sys.argv[2]

fig = plt.figure (1)
plot1 = plot_dist(graph_1_adrs, 'g')

# file 2
if(len(sys.argv) == 6):
    # print('double')
    double = True
    graph_2_adrs = sys.argv[4]
    plot2 = plot_dist(graph_2_adrs, 'r')

graph_1 = sys.argv[3]
graph_2 = sys.argv[5]


plt.xscale ('log')
plt.yscale ('log')
plt.ylabel('Normalized frequency of nodes')
plt.xlabel('Orbit '+ orbit + ' count')
plt.title('Count distribution of orbit' + orbit)
if double:
    plt.legend([plot1,plot2],[graph_1, graph_2])
else:
    plt.legend([plot1],[graph_1])
plt.savefig('./orbit_'+orbit+'_count_dist_'+ graph_1 + '_' + graph_2 + '.png', bbox_inches='tight', dpi=400)
