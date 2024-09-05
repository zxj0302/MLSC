import graph_tools
import triangle_counters
import four_vertex_counters
import five_vertex_counters
import sys


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


# Usage:
# python tester_five.py <GRAPH NAME>

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
        #print(node,nbr)
        #orbital_edge[(node, nbr)] = [0.0] * 73  # initalize each edge orbital starting count from 0
        #orbital_edge[(nbr, node)] = [0.0] * 73  # initalize each edge orbital starting count from 0

print(orbital_vertex['2'])

#graphs of size three:
orbital_vertex = triangle_counters.triangle_orbital_vertex(G, orbital_vertex)
#orbital_edge = triangle_counters.triangle_orbital_edge(G, orbital_edge)
orbital_vertex = four_vertex_counters.four_vertex_orbital(G,orbital_vertex)
#orbital_edge = four_vertex_counters.four_vertex_edge(G,orbital_edge)
orbital_vertex = five_vertex_counters.five_orbitals(G, orbital_vertex)

orbital_vertex = devide_orbital_vertex(orbital_vertex)
#orbital_edge = devide_orbital_edge(orbital_edge)
#count = 66

for count in range(0,73):
	print(count,
		"0:", orbital_vertex['0'][count],
		" 1:", orbital_vertex['1'][count],
		" 2:", orbital_vertex['2'][count],
		" 3:", orbital_vertex['3'][count],
		" 4:", orbital_vertex['4'][count],
		" 5:", orbital_vertex['5'][count])

#for count in range(len(orbital_vertex['0'])):
    #continue
 #   print(count, "0:", orbital_vertex['0'][count], " 1:", orbital_vertex['1'][count], " 2:", orbital_vertex['2'][count], " 3:", orbital_vertex['3'][count], " 4:", orbital_vertex['4'][count], " 5:", orbital_vertex['5'][count])





# four_counts = four_vertex_counters.four_vertex_count(G,'','',False)
# all_counts = five_vertex_counters.all_five_vertex(G,'','',False)
# all_disc_counts = five_vertex_counters.all_five_disconnected(G,'','')





# print("------------------------")
# print(long(four_counts[0]), "3-stars")
# print(long(four_counts[1]), "3-paths")
# print(long(four_counts[2]), "tailed-triangles")
# print(long(four_counts[3]), "4-cycles")
# print(long(four_counts[4]), "diamonds")
# print(long(four_counts[5]), "4-cliques")
#
# print("---------------")
# print(long(all_disc_counts[0]), "Ind set")
# print(long(all_disc_counts[1]), "Only edge")
# print(long(all_disc_counts[2]), "Matching")
# print(long(all_disc_counts[3]), "Only wedge")
# print(long(all_disc_counts[4]), "Only triangle")
# print(long(all_disc_counts[5]), "Only 3-star")
# print(long(all_disc_counts[6]), "Only 3-path")
# print(long(all_disc_counts[7]), "Only Tailed tri")
# print(long(all_disc_counts[8]), "Only 4-cycle")
# print(long(all_disc_counts[9]), "Only Diamond")
# print(long(all_disc_counts[10]), "Only 4-clique")
# print(long(all_disc_counts[11]), "Wedge+edge")
# print(long(all_disc_counts[12]), "Triangle+edge")
#
#
#
# print("----------------")
# print(all_counts[0], "stars")
# print(all_counts[1], "prongs")
# print(all_counts[2], "four paths")
# print(all_counts[3], "fork-tailed triangle")
# print(all_counts[4], "long-tailed triangle")
# print(all_counts[5], "double-tailed triangle")
# print(all_counts[6], "tailed 4-cycles")
# print(all_counts[7], "five cycles")
# print(all_counts[8], "hourglasses")
# print(all_counts[9], "cobras")
# print(all_counts[10], "stingrays")
# print(all_counts[11], "hatted 4-cycles")
# print(all_counts[12], "three wedge collisions")
# print(all_counts[13], "three triangles collisions")
# print(all_counts[14], "tailed 4-cliques")
# print(all_counts[15], "triangle strips")
# print(all_counts[16], "diamond-wedge collisions")
# print(all_counts[17], "wheels")
# print(all_counts[18], "hatted 4-cliques")
# print(all_counts[19], "almost cliques")
# print(all_counts[20], "five cliques")
#
# f = open("nums_test.txt","w")
#
# f.write(str(int(four_counts[0]))+"\n")
# f.write(str(int(four_counts[1]))+"\n")
# f.write(str(int(four_counts[2]))+"\n")
# f.write(str(int(four_counts[3]))+"\n")
# f.write(str(int(four_counts[4]))+"\n")
# f.write(str(int(four_counts[5]))+"\n")
# f.write(str(int(all_counts[0]))+"\n")
# f.write(str(int(all_counts[1]))+"\n")
# f.write(str(int(all_counts[2]))+"\n")
# f.write(str(int(all_counts[3]))+"\n")
# f.write(str(int(all_counts[4]))+"\n")
# f.write(str(int(all_counts[5]))+"\n")
# f.write(str(int(all_counts[6]))+"\n")
# f.write(str(int(all_counts[7]))+"\n")
# f.write(str(int(all_counts[8]))+"\n")
# f.write(str(int(all_counts[9]))+"\n")
# f.write(str(int(all_counts[10]))+"\n")
# f.write(str(int(all_counts[11]))+"\n")
# f.write(str(int(all_counts[12]))+"\n")
# f.write(str(int(all_counts[13]))+"\n")
# f.write(str(int(all_counts[14]))+"\n")
# f.write(str(int(all_counts[15]))+"\n")
# f.write(str(int(all_counts[16]))+"\n")
# f.write(str(int(all_counts[17]))+"\n")
# f.write(str(int(all_counts[18]))+"\n")
# f.write(str(int(all_counts[19]))+"\n")
# f.write(str(int(all_counts[20]))+"\n")







