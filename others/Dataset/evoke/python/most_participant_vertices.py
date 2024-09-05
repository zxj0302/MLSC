# This is a script to find the vertices which have the most number of  participation in an orbit.
# USAGE:
#		python3 most_participation_vertex.py <ORBIT NUMBER> <ORIGINAL UNPARSED CITATION NETWORK DATA> <SANITIZED GRAPH ADDRESS> <GRAPH SANITIZATION MAP ADDRESS> 
#		GRAPH SANITIZATION MAP ADDRESS:
#		Is the map from id the id given to the vertices in the sanitized version of the graph to the actual id of the vertices in the original version of the graph
# The script prints the original id of the discovered vertices from the original unparsed graph data and their meta-data to the following file:
#		<ORBIT NUMBER>_most_participant_vertices_info.txt

# Noujan Pashanasangi, June 2019
#
#

import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import json

total_counts = 0


if len(sys.argv) < 5: # args not correct
    print('USAGE: python3 plot_dd.py  <ORBIT NUMBER> <ORBIT COUNTS OUTPUT> <ORIGINAL UNPARSED CITATION NETWORK DATA>  <GRAPH SANITIZATION MAP ADDRESS>')
    sys.exit(-1)

orbit = sys.argv[1]

orbit_counts_addr = sys.argv[2]

graph_meta_data_addr = sys.argv[3]

graph_map_addr = sys.argv[4]


orbit_counts = open(orbit_counts_addr, 'r')

max_participation = 0
top_vertices = list()
top_vertices_original_id = list()



vertex_index = 0
orbit_index = int(float(orbit))

for line in orbit_counts.readlines():	
    tokens = line.split(' ')
    if(len(tokens) < 70):
        print('orbits less than 72')
        print(tokens)
        print(len(tokens))
    current_count = int(float(tokens[orbit_index]))
    if current_count > max_participation:
    	max_participation = current_count

print(max_participation)
orbit_counts.close()
orbit_counts = open(orbit_counts_addr, 'r')

for line in orbit_counts.readlines():	
    tokens = line.split(' ')
    current_count = int(float(tokens[orbit_index]))
    if current_count == max_participation:
    	top_vertices.append(vertex_index)
    vertex_index = vertex_index + 1


json_file = open(graph_map_addr, 'r')

mapping = json.load(json_file)

#inversing the mapping
inv_mapping = {v: k for k, v in mapping.items()}

print(top_vertices)

top_vertices_original_id = [inv_mapping[v] for v in top_vertices]

print(top_vertices_original_id)

top_vertices_original_id = set(top_vertices_original_id)


top_vertex_index = 0

graph_meta_data = open(graph_meta_data_addr, 'r')
top_vertices_info = open(orbit + '_most_participant_vertices_info.txt', 'w')


current_vertex_meta_data = list()

lines = graph_meta_data.readlines()

potential_top = True
is_top = False

for line in lines:
	if line.startswith('#') is False:
		continue
	if line.startswith('#index'):
		index = line[6:-1]
		if index in top_vertices_original_id:
			is_top = True
		else:
			potential_top = False
			continue
	elif line.startswith('#*'):
		if is_top:
			for data_line in current_vertex_meta_data:
				top_vertices_info.write(data_line)
			top_vertices_info.write('\n')
		is_top = False
		potential_top = True
		current_vertex_meta_data.clear()
		current_vertex_meta_data.append(line)	
	elif potential_top:
		current_vertex_meta_data.append(line)
	


top_vertices_info.close()
graph_meta_data.close()
json_file.close()
orbit_counts.close()


















