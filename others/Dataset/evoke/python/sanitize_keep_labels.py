import os
import sys
import graph_tools

# Takes as input a file with edges list as pairs of strings, and creates
# the proper input for escape. 
#
# Usage: sanitize.py directory file1 file2 file3 ...
# where file1, file2, etc. are in directory.
# Output: file1.edges file2.edges, etc. in proper format, all placed in directory.
# This format has the first line with the number of nodes and edges.
# Each line has a distinct edge with node labels as ints starting from 0.
#


if len(sys.argv) < 3:
    print('Usage: python sanitize.py <DIR> -d <FILE1> <FILE2>... ') # -d is optional flag, decrease the node numbers by 1
    sys.exit

# decrease = False

file_start = 2
# if '-d' in sys.argv:
#     decrease = True
#     file_start = 3

os.chdir(sys.argv[1])

min_node = -1
max_node = -1

for i in range(file_start,len(sys.argv)):   # Loop over arguments
    fname = sys.argv[i]
    f_input = open(fname,'r')   # File pointer to input
    list_lines = f_input.readlines()  # List of lines
    
    names = fname.split('.')
    out_name = names[0] + '.edges'  # Output file name
    f_output = open(out_name,'w')   # Output file ptr

    G = graph_tools.graph()

    line_num = 0

    ind = 0
    #mapping = dict()
    for each_line in list_lines:
        line_num = line_num+1
#         if line_num == 1:   # Ignore first line
#             continue
        line = each_line.strip()
        tokens = line.split()   # Getting list of node names and edge attributes
        if tokens[0] == tokens[1]: # Removing self-loop
            continue
        if tokens[0] == "#":
        	continue
        # if tokens[0] in mapping:    # If first node name is already mapped
        #     node1 = mapping[tokens[0]]  # Get numeric map value of first node name
        # else:
        #     mapping[tokens[0]] = ind    # Introduce new node to mapping
        #     node1 = ind
        #     ind = ind+1                 # Increment the index

        # if tokens[1] in mapping:    # Perform same task for second node name
        #     node2 = mapping[tokens[1]]
        # else:
        #     mapping[tokens[1]] = ind
        #     node2 = ind
        #     ind = ind+1
        #print(tokens[0], tokens[1])
        
        # if decrease:
        #     G.Add_und_edge(str(int(tokens[0])-1),str(int(tokens[1])-1))
        # else:
        #     G.Add_und_edge(tokens[0],tokens[1])

        G.Add_und_edge(tokens[0],tokens[1])

        if min_node == -1 or tokens[0] < min_node:
            min_node = tokens[0]
        if min_node == -1 or tokens[1] < min_node:
            min_node = tokens[1]

        if tokens[0] > max_node:
            max_node = tokens[0]
        if tokens[1] > max_node:
            max_node = tokens[1]


    
    n = int(max_node) + 1
    f_output.write(str(n)+' ')
    m = 0;
    for node in G.vertices:
        m = m + G.degrees[node]
    m = m/2
    f_output.write(str(m)+'\n')

    print n, "vertices and", m, "edges"

    for node in sorted(G.vertices, key = int):
        for nbr in G.adj_list[node]:
            if node == n:
                print node, "node out of range"
            if nbr == n:
                print nbr, "nbr out of range"
            if node < nbr:
                f_output.write(str(node)+' '+str(nbr)+'\n')

    f_input.close()
    f_output.close()
    
