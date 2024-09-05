import os
import numpy as np
import math
import sys

import os
import numpy as np
import sys
import math

import graph_tools
import triangle_counters
import four_vertex_counters
import five_vertex_counters

n = int(sys.argv[1])
counter = 0


adj = np.zeros((n,n))

with open("test_out.txt", 'w') as test_out_file:
    test_out_file.write("testing all the graphs with " + str(n) + " nodes\n")

def print_graph(n):
    global counter
    m = 0
    for i in range(0, n):
        for j in range(i+1, n):
            m = m + int(adj[i][j])
    with open("graph_"+str(counter)+".edges", 'w') as file:
        file.write(str(n)+ " " + str(m))
        file.write("\n")
        for i in range(0, n):
            for j in range(i+1, n):
                if adj[i][j] == 1:
                    file.write(str(i))
                    file.write(" ")
                    file.write(str(j))
                    file.write("\n")
    os.system("python3 test.py graph_"+str(counter)+".edges 5 -i -p" + " && " + "rm graph_"+str(counter)+".edges")
    # os.system("rm graph_"+str(counter)+".edges")
    counter = counter + 1



def call_next(row, col, n):
    if row == n-1 and col == n-1 :
        #print("print graph")    
        print_graph(n)
    elif col < n-1:
        creat_graph(row, col+1, n)
    elif col == n-1:
        creat_graph(row + 1, 0, n)



def creat_graph(row, col, n):
    #print("in create graph")
    #print(row, col)
    if row == col:
        adj[row][col] = 0
        call_next(row, col, n)
    elif row > col:
        adj[row][col] = adj[col][row]
        call_next(row, col, n)
    else:
        adj[row][col] = 1
        call_next(row, col, n)
        adj[row][col] = 0
        call_next(row, col, n)


creat_graph(0,0,n)
