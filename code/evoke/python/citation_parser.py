# This is a script to parse citation networks from "https://aminer.org/citation".
# USAGE:
#		python3 citation_parser.py <CITATION NETWORK DATA> <OUTPUT GRAPH FILE ADDRESS>
# The script creates the graph from the citation network raw data, and writes the list of edges of the graph to the output file. The output file should be sanitized later.
# Noujan Pashanasangi, June 2019
#

import numpy as np
import sys
import graph_tools


index = str(0)

input_file = sys.argv[1]
output_file = sys.argv[2]


G = graph_tools.graph()
n = 0

with open(input_file, 'r') as f_in:
	# 
	lines = f_in.readlines()
	for line in lines:
		if line.startswith('#index'):
			index = line[6:-1]
			n = max(n, int(index))
		if line.startswith("#%"):
			neighbor = line[2:-1]
			n = max(n, int(neighbor))
			if neighbor != index:
				G.Add_und_edge(index,neighbor)
			# f_out.write(index+" "+neighbor+"\n")
n = n+1
with open(output_file, 'w') as f_out:
	# n = len(G.vertices)
	# f_out.write(str(n)+' ')
	m = 0;
	for node in G.vertices:
	    m = m + G.degrees[node]
	m = m/2
	m = int(m)
	# f_out.write(str(m)+'\n')

	# print n, "vertices and", m, "edges"

	for node in G.vertices:
	    for nbr in G.adj_list[node]:
	        if int(node) >= n:
	            print( node, "node out of range")
	        if int(nbr) >= n:
	            print( nbr, "nbr out of range")
	        if node < nbr:
	            f_out.write(str(node)+' '+str(nbr)+'\n')
print(n, " nodes and ", m, " edges")