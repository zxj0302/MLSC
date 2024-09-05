import os
import numpy as np
import sys
import math

import graph_tools
import triangle_counters
import four_vertex_counters
import five_vertex_counters

# Usage :
# python3 test.py <Graph Name> <number of nodes of the orbits to count> -i -p

if len(sys.argv) < 3:
    print( 'Format: python3 orbit_counts.py <PATH FOR INPUT> <DESIRED PATTERN SIZE> <OPTIONS>\n DESIRED PATTERN SIZE: 3, 4, or 5\n OPTIONS:\n \t -i: for integer outputs, useful in looking at small graphs\n \t -p: for printing orbit counts for all the nodes' )
    sys.exit()

integral = False
orbitPrint = False

if '-i' in sys.argv:
    integral = True

if '-p' in sys.argv:
    orbitPrint = True

pattern = int(sys.argv[2])  # size of desired patterns

if pattern == 3:   # calling appropriate executable, output is in out.txt
    os.system('../exe/count_orbit_three '+sys.argv[1])
elif pattern == 4:
    os.system('../exe/count_orbit_four '+sys.argv[1])
elif pattern == 5:
    os.system('../exe/count_orbit_five '+sys.argv[1])
else:
    print('Incorrect format: Desired pattern size must be 3, 4, or 5')
    sys.exit()





#########################################################################################


# clean up deviding
def devide_orbital_vertex(orbital_vertex):
    for node in orbital_vertex:
        for orbital_count in range(len(orbital_vertex[node])):
            if orbital_count == 0:
                orbital_vertex[node][orbital_count] //= 2

            elif orbital_count == 3:
                orbital_vertex[node][orbital_count] //= 3
            elif orbital_count == 4:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 5:
                orbital_vertex[node][orbital_count] //= 2

            elif orbital_count == 8:
                orbital_vertex[node][orbital_count] //= 4

            elif orbital_count == 12:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 13:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 14:
                orbital_vertex[node][orbital_count] //= 4

            elif orbital_count == 22:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 23:
                orbital_vertex[node][orbital_count] //= 2

            elif orbital_count == 24:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 25:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 26:
                orbital_vertex[node][orbital_count] //= 2
            # elif orbital_count == 29:
            #     orbital_vertex[node][orbital_count] //= 2
            # elif orbital_count == 31:
            #     orbital_vertex[node][orbital_count] //= 2
            # elif orbital_count == 32:
            #     orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 34:
                orbital_vertex[node][orbital_count] //= 5
            # elif orbital_count == 37:
            #     orbital_vertex[node][orbital_count] //= 2



            elif orbital_count == 49:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 50:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 51:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 52:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 53:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 54:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 55:
                orbital_vertex[node][orbital_count] //= 2
            # elif orbital_count == 57:
            #     orbital_vertex[node][orbital_count] //= 3
            # elif orbital_count == 59:
            #     orbital_vertex[node][orbital_count] //= 2
            # elif orbital_count == 60:
            #     orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 62:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 63:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 64:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 65:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 66:
                orbital_vertex[node][orbital_count] //= 2
            elif orbital_count == 67:
                orbital_vertex[node][orbital_count] //= 2
            # elif orbital_count == 68:
            #     orbital_vertex[node][orbital_count] //= 4
            elif orbital_count == 70:
                orbital_vertex[node][orbital_count] //= 3
            elif orbital_count == 71:
                orbital_vertex[node][orbital_count] //= 3
            elif orbital_count == 72:
                orbital_vertex[node][orbital_count] //= 5
            else:
                continue

    return orbital_vertex


def devide_orbital_edge(orbital_edge):
    for node in G.vertices:
        for nbr in G.adj_list[node]:
            for orbital_count in range(len(orbital_edge[(node, nbr)])):
                if orbital_count == 0:
                    orbital_edge[(node, nbr)][orbital_count] //= 2
                elif orbital_count == 1:
                    orbital_edge[(node, nbr)][orbital_count] //= 3
                elif orbital_count == 2:
                    orbital_edge[(node, nbr)][orbital_count] //= 2
                elif orbital_count == 4:
                    orbital_edge[(node, nbr)][orbital_count] //= 3
                elif orbital_count == 5:
                    orbital_edge[(node, nbr)][orbital_count] //= 4
                elif orbital_count == 8:
                    orbital_edge[(node, nbr)][orbital_count] //= 2
                elif orbital_count == 9:
                    orbital_edge[(node, nbr)][orbital_count] //= 4
                elif orbital_count == 11:
                    orbital_edge[(node, nbr)][orbital_count] //= 6

            for orbital_count in range(len(orbital_edge[(nbr, node)])):
                if orbital_count == 0:
                    orbital_edge[(nbr, node)][orbital_count] //= 2
                elif orbital_count == 1:
                    orbital_edge[(nbr, node)][orbital_count] //= 3
                elif orbital_count == 2:
                    orbital_edge[(nbr, node)][orbital_count] //= 2
                elif orbital_count == 4:
                    orbital_edge[(nbr, node)][orbital_count] //= 3
                elif orbital_count == 5:
                    orbital_edge[(nbr, node)][orbital_count] //= 4
                elif orbital_count == 8:
                    orbital_edge[(nbr, node)][orbital_count] //= 2
                elif orbital_count == 9:
                    orbital_edge[(nbr, node)][orbital_count] //= 4
                elif orbital_count == 11:
                    orbital_edge[(nbr, node)][orbital_count] //= 6

    return orbital_edge


name = sys.argv[1]

G = graph_tools.graph()
G.Read_edges(name)

# order = G.DegenOrdering()
# G = G.Orient(order)

#print(G.adj_list['0'])

#initalize needed dictionaries to store counts of the orbitals
orbital_vertex = {}
orbital_edge = {}

#print(DAG.vertices)
for node in G.vertices:
    continue
    #print(node, G.adj_list[node])

#print('next')
for node in G.vertices:
    orbital_vertex[node] = [0.0] * 73  # intialize each vertex orbital starting count from 0
    for nbr in G.adj_list[node]:
        continue

#graphs of size three:
orbital_vertex = triangle_counters.triangle_orbital_vertex(G, orbital_vertex)
#orbital_edge = triangle_counters.triangle_orbital_edge(G, orbital_edge)
orbital_vertex = four_vertex_counters.four_vertex_orbital(G,orbital_vertex)
#orbital_edge = four_vertex_counters.four_vertex_edge(G,orbital_edge)
orbital_vertex = five_vertex_counters.five_orbitals(G, orbital_vertex)

orbital_vertex = devide_orbital_vertex(orbital_vertex)
#orbital_edge = devide_orbital_edge(orbital_edge)


if orbitPrint:
    f_out = open('out.txt','r')

    node_orbits = [[int(x) for x in line.split()] for line in f_out]

    check = 0
    for i in range(len(node_orbits)):
        verifier_counts = [int(orbital_vertex[node][i]) for node in sorted(G.vertices, key=int)]
        if(node_orbits[i] != verifier_counts):
            with open("test_out.txt", 'a+') as test_out_file:
                check = 1
                test_out_file.write(str(i) + ":\n")
                test_out_file.write("node orbits: [")
                test_out_file.write(" ".join([str(x) for x in node_orbits[i]]))
                test_out_file.write("]\n")
                test_out_file.write("verifier counts: [")
                test_out_file.write(" ".join([str(x) for x in verifier_counts]))
                test_out_file.write("]\n")

    if check == 1:
        with open("test_out.txt", 'a+') as test_out_file:
            test_out_file.write(sys.argv[1]+"\n\n")

    f_out.close()

