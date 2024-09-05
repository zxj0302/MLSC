import os
import numpy as np
import math
import sys
import random
from datetime import datetime


# Run using "python3 ./graph_sample_test.py n num_sample"
# where n is the number of nodes of the graphs you want to sample
# and num_sample is the number of the graphs you want to use as a test

def print_graph(n):
    global counter
    #print("in print_graph")
    m = 0
    for i in range(0, n):
        for j in range(i+1, n):
            m = m + int(adj[i][j])
    with open("graph_"+str(counter)+".edges", 'w') as file:
        #print("in file writing", counter)
        #print(int(m))
        file.write(str(n)+ " " + str(m))
        file.write("\n")
        for i in range(0, n):
            for j in range(i+1, n):
                if adj[i][j] == 1:
                    file.write(str(i))
                    file.write(" ")
                    file.write(str(j))
                    file.write("\n")
    # os.system("python3 test.py graph_"+str(counter)+".edges 5 -i -p")
    os.system("python3 test.py graph_"+str(counter)+".edges 5 -i -p" + " && " + "rm graph_"+str(counter)+".edges")
    # os.system("rm graph_"+str(counter)+".edges")
    counter = counter + 1


random.seed(datetime.now())

n = int(sys.argv[1])
num_samples = int(sys.argv[2])
counter = 0

adj = np.zeros((n,n))
edge_mark = set()


with open("test_out.txt", 'w') as test_out_file:
    test_out_file.write("testing " + str(num_samples) + " sampled graphs with " + str(n) + " nodes\n")


for smaple in range(num_samples):
    adj = np.zeros((n,n))
    edge_mark.clear()
    m = random.randint(0, (n * (n - 1) )/ 2)
    print("m: " + str(m))

    for e in range(m):
        
        a = -1
        b = -1

        while a == b or (a,b) in edge_mark:
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a > b:
                a, b = b, a
        
        edge_mark.add((a,b))
        adj[a][b] = 1
    print("m: " + str(m))
    print_graph(n)



