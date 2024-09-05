import itertools
import sys
import triangle_counters
import four_vertex_counters
import numpy as np


def all_five_vertex(G,fname='',gname='',to_list=False):
    five_counts = four_path_based(G,'','',False)
    prong_count = prong(G,'','',False)
    wedge_stuff = wedge_collision(G,'','',False)
    stars = fourstar(G,'','',False)

    all_counts = []
    all_counts.append(stars)
    all_counts.append(prong_count)
    all_counts.append(five_counts[4])
    all_counts.append(five_counts[7])
    all_counts.append(five_counts[6])
    all_counts.append(five_counts[5])
    all_counts.append(five_counts[0])
    all_counts.append(five_counts[11])
    all_counts.append(five_counts[8])
    all_counts.append(five_counts[9])
    all_counts.append(five_counts[10])
    all_counts.append(five_counts[1])
    all_counts.append(wedge_stuff[0])
    all_counts.append(wedge_stuff[1])
    all_counts.append(five_counts[2])
    all_counts.append(five_counts[12])
    all_counts.append(wedge_stuff[3])
    all_counts.append(wedge_stuff[2])
    all_counts.append(five_counts[3])
    all_counts.append(wedge_stuff[4])
    all_counts.append(wedge_stuff[5])

    return all_counts

def all_five_disconnected(G,fname='',gname=''):
    all_counts = []
    size_info = G.Size()
    n = size_info[0]
    m = size_info[1]/2
    w = size_info[2]
    
    order = G.DegenOrdering()
    DG = G.Orient(order)
    
    tri_info = triangle_counters.triangle_info(DG)
    t = sum(tri_info[0].values())/3
    
    four_counts = four_vertex_counters.four_vertex_count(G,'','',False)
    
    all_counts.append((n*(n-1)*(n-2)*(n-3)*(n-4))/120)
    all_counts.append(m*((n-2)*(n-3)*(n-4))/6)
    all_counts.append(((m*(m-1))/2 - w)*(n-4))
    all_counts.append(w*((n-3)*(n-4))/2)
    all_counts.append(t*((n-3)*(n-4))/2)
    all_counts.append(four_counts[0]*(n-4));   
    all_counts.append(four_counts[1]*(n-4));   
    all_counts.append(four_counts[2]*(n-4));   
    all_counts.append(four_counts[3]*(n-4));   
    all_counts.append(four_counts[4]*(n-4));   
    all_counts.append(four_counts[5]*(n-4));   
    all_counts.append(w*(m-2) - 3*t - 3*four_counts[0] - 2*four_counts[1])
    all_counts.append(t*(m-3) - four_counts[2])

    return all_counts

def fourstar(G,fname='',gname='',to_list=False):
    stars = 0
    for node1 in G.vertices:
        deg = len(G.adj_list[node1])
        stars = stars + (deg*(deg-1)*(deg-2)*(deg-3))/24
    return stars

def wedge_collision(G,fname='',gname='',to_list=False):
    three_wed = 0
    diamond_wed = 0
    three_tri = 0
    wheel = 0
    five_clique = 0
    almost_clique = 0

    wedges = dict()
    for node1 in G.vertices:
        for (nbr1, nbr2) in itertools.combinations(G.adj_list[node1],2):
            if nbr2 > nbr1:
                swap = nbr1
                nbr1 = nbr2
                nbr2 = swap

#             print 'outer', node1, nbr1, nbr2
            if (nbr1,nbr2) not in wedges:
                wedges[(nbr1,nbr2)] = set({node1})
            else:
                wedges[(nbr1,nbr2)].add(node1)
#                 print 'adding', nbr1, nbr2, wedges[(nbr1,nbr2)]

    for pair in wedges.keys():
        num = len(wedges[pair])
        three_wed = three_wed + num*(num-1)*(num-2)/6

        if to_list:
            print(pair, wedges[pair])

        if pair[0] in G.adj_list[pair[1]]:
            connected = True
        else:
            connected = False

        for (nbr1, nbr2, nbr3) in itertools.combinations(wedges[pair],3):
            num_edges = 0
            if to_list:
                print(pair, nbr1, nbr2, nbr3)
            if nbr1 in G.adj_list[nbr2]:
                num_edges += 1
            if nbr1 in G.adj_list[nbr3]:
                num_edges += 1
            if nbr2 in G.adj_list[nbr3]:
                num_edges += 1



            diamond_wed += num_edges
            wheel = wheel+ (num_edges*(num_edges-1))/2
            almost_clique += (num_edges*(num_edges-1)*(num_edges-2))/6
            
            if connected:
                three_tri += 1
                almost_clique += (num_edges*(num_edges-1))/2
                five_clique += (num_edges*(num_edges-1)*(num_edges-2))/6
#                 if num_edges == 3:
#                     dummy = [int(pair[0]), int(pair[1]), int(nbr1), int(nbr2), int(nbr3)]
#                     dummy = sorted(dummy)
#                     print "found from",pair[0],pair[1]
#                     print "clique", dummy[0], dummy[1], dummy[2], dummy[3], dummy[4]

#             print connected, num_edges

           
    wheel = wheel/2
    five_clique = five_clique/10
    almost_clique = almost_clique/4  

    return (three_wed, three_tri, wheel, diamond_wed, almost_clique, five_clique)

def prong(G,fname='',gname='',to_list=False):
    prong = 0

    for node1 in G.vertices:
        for node2 in G.adj_list[node1]:
            for (nbr1, nbr2) in itertools.combinations(G.adj_list[node2],2):
                if nbr1 == node1 or nbr2 == node1:
                    continue
                for node3 in G.adj_list[node1]:
                    if node3 == node2 or node3 == nbr1 or node3 == nbr2:
                        continue
                    prong = prong+1
    return prong
  
def graph_definitions(vector1, vector2, vector3, orbital_vertex, DG):
    #vector 1 is the count of each deg of node ie: [a,b,c,d,e] is the count of node1:a, node2:b, node3:c, node4:d, node4:e
    #vector 2 is the count of each deg ie [1,2,3,4,5] or how many nodes with 1 edge, how nodes with 2 edges, etc.
    #vector 3 is the actual nodes themselves, ie [node1, node2, node3, node4, node5]
    #orbital_vertex is the dict that stores counts of the orbitals for each node
    #DG is the graph
    graph = {}
    graph[(2,0,0,0,0)] = [-1,0,0,0,0,0] #graph 0    key: -1 = orbital 0 everything else is same
    graph[(2,1,0,0,0)] = [1,2,0,0,0,0] #graph 1
    graph[(0,3,0,0,0)] = [0,3,0,0,0,0] #graph 2
    graph[(2,2,0,0,0)] = [4,5,0,0,0,0] #graph 3
    graph[(3,0,1,0,0)] = [6,0,7,0,0,0] #graph 4
    graph[(0,4,0,0,0)] = [0,8,0,0,0,0] #graph 5
    graph[(1,2,1,0,0)] = [9,10,11,0,0,0] #graph 6
    graph[(0,2,2,0,0)] = [0,12,13,0,0,0] #graph 7
    graph[(0,0,4,0,0)] = [0,0,14,0,0,0] #graph 8
    graph[(2,3,0,0,0)] = [15,[16,17],0,0,9,0] #graph 9 #special 16,17
    graph[(3,1,1,0,0)] = [[18,19],20,21,0,10,0] #graph 10 # #special 18,19
    graph[(4,0,0,1,0)] = [22,0,0,23,0,0] #graph 11
    graph[(2,1,2,0,0)] = [24,25,26,0,0,0] #graph 12
    graph[(1,3,1,0,0)] = [27,[28,29],30,0,13,1] #graph 13 #special 28,29 #duplicate with graph 16
    graph[(2,2,0,1,0)] = [31,32,0,33,0,0] #graph 14
    graph[(0,5,0,0,0)] = [0,34,0,0,0,0] #graph 15
    #graph[[1,3,1,0,0]] = [35,[36,37],38,0,16,1] #graph 16 #special 36,37 #duplicate with graph 13
    graph[(1,2,1,1,0)] = [39,40,41,42,0,0] #graph 17
    graph[(0,4,0,1,0)] = [0,43,0,44,0,0] #graph 18
    graph[(1,1,3,0,0)] = [45,46,[47,48],0,19,0] #graph 19 #special 47,48
    graph[(0,3,2,0,0)] = [0,49,50,0,21,1] #graph 20 #duplicate with graph 21
    #graph[[0,3,2,0,0]] = [0,[51,52],53,0,21,1] #graph 21 #special 51,52 #duplicate with graph 20
    graph[(0,3,0,2,0)] = [0,54,0,55,0,0] #graph 22
    graph[(1,0,3,1,0)] = [56,0,57,58,0,0] #graph 23
    graph[(0,2,2,1,0)] = [0,59,60,61,0,0] #graph 24
    graph[(0,1,4,0,0)] = [0,62,[63,64],0,25,0] #graph 25 #special 63,64
    graph[(0,1,2,2,0)] = [0,65,66,67,0,0] #graph 26
    graph[(0,0,4,1,0)] = [0,0,68,69,0,0] #graph 27
    graph[(0,0,2,3,0)] = [0,0,70,71,0,0] #graph 28
    graph[(0,0,0,5,0)] = [0,0,0,72,0,0] #graph 29

    #get orbital count graph graph
    orbital = graph[vector2]
    if orbital[5] == 0:
        if orbital[4] == 0:
            #case where no special and duplicates
            for node in range(len(vector1)):
                orb = orbital[vector1[node]-1]
                if orb == -1:
                    orbital_vertex[vector3[node]][0] += 1
                if orb > 0:
                    orbital_vertex[vector3[node]][orb] += 1
        else:
            #case where no duplicates but specials
            #graph 9
            if orbital[4] == 9:
                orbital_vertex[vector3[0]][17] += 1
                orbital_vertex[vector3[1]][16] += 1
                orbital_vertex[vector3[2]][16] += 1
                orbital_vertex[vector3[3]][15] += 1
                orbital_vertex[vector3[4]][15] += 1

            #graph 10
            elif orbital[4]==10:
                deg2 = vector1.index(2)
                deg1 = [index for index, value in enumerate(vector1) if value == 1]
                for index in deg1:
                    if index != 4:
                        orbital_vertex[vector3[index]][19] += 1
                orbital_vertex[vector3[0]][21] += 1
                orbital_vertex[vector3[deg2]][20] += 1
                orbital_vertex[vector3[4]][18] += 1

            elif orbital[4]==19:
                deg3 = [index for index, value in enumerate(vector1) if value == 3]
                deg1 = vector1.index(1)
                for index in deg3:
                    if DG.isEdge(vector3[index], vector3[deg1]):
                        orbital_vertex[vector3[index]][47] += 1
                        orbital_vertex[vector3[deg1]][45] += 1
                        for index2 in deg3:
                            if index2 != index:
                                orbital_vertex[vector3[index2]][48] += 1
                        break
                deg2 = vector1.index(2)
                orbital_vertex[vector3[deg2]][46] += 1

            elif orbital[4]==25:
                deg4 = [index for index, value in enumerate(vector1) if value == 4]
                deg3 = [index for index, value in enumerate(vector1) if value == 3]
                deg2 = vector1.index(2)
                for index in deg3:
                    # orbital_vertex[vector3[index]][63] += 1
                    orbital_vertex[vector3[index]][64] += 1
                for index in deg4:
                    # orbital_vertex[vector3[index]][64] += 1
                    orbital_vertex[vector3[index]][63] += 1
                orbital_vertex[vector3[deg2]][62] += 1
            else:
                Exception("graph is out of bounds")
    else:
        # case where duplicates and specials
        if orbital[4] == 13:
            deg3 = vector1.index(3)
            deg1 = vector1.index(1)
            deg2 = [index for index, value in enumerate(vector1) if value == 2]
            if 5 in vector1:
                deg5 = vector1.index(5)
                # graph 16
                for index in deg2:
                    orbital_vertex[vector3[index]][37] += 1
                orbital_vertex[vector3[deg5]][36] += 1
                orbital_vertex[vector3[deg3]][38] += 1
                orbital_vertex[vector3[deg1]][35] += 1
            else:
                #graph 13
                deg4 = vector1.index(4)
                for index in deg2:
                    orbital_vertex[vector3[index]][29] += 1
                orbital_vertex[vector3[deg3]][30] += 1
                orbital_vertex[vector3[deg1]][27] += 1
                orbital_vertex[vector3[deg4]][28] += 1

        elif orbital[4] == 21:
            if 2 in vector1:
                #graph 21
                deg4 = vector1.index(4)
                deg3 = [index for index, value in enumerate(vector1) if value == 3]
                deg2 = [index for index, value in enumerate(vector1) if value == 2]
                for index in deg2:
                    orbital_vertex[vector3[index]][51] += 1
                for index in deg3:
                    orbital_vertex[vector3[index]][53] += 1
                orbital_vertex[vector3[deg4]][52] += 1

            else:
                # graph 20
                deg3 = [index for index, value in enumerate(vector1) if value == 4]
                deg2 = [index for index, value in enumerate(vector1) if value == 5]
                for index in deg3:
                    orbital_vertex[vector3[index]][50] += 1
                for index in deg2:
                    orbital_vertex[vector3[index]][49] += 1
        else:
            Exception("graph is out of bounds")


def five_orbitals(DG, orbital_vertex):
    for node1 in DG.vertices:  # Loop over all nodes
        for (node2, node3, node4, node5) in itertools.combinations(DG.adj_list[node1],4):# Loop over all three-pairs of neighbors of node1
            #print(node2, node3, node4, node5)
            #graph 11
            graph_definitions([4, 1, 1, 1, 1], (4, 0, 0, 1, 0), [node1, node2, node3, node4, node5], orbital_vertex, DG)

            #graph 14
            if DG.isEdge(node2, node3):
                graph_definitions([4, 2, 2, 1, 1], (2, 2, 0, 1, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)
            if DG.isEdge(node2, node4):
                graph_definitions([4, 2, 1, 2, 1], (2, 2, 0, 1, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)
            if DG.isEdge(node2, node5):
                graph_definitions([4, 2, 1, 1, 2], (2, 2, 0, 1, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)
            if DG.isEdge(node3, node4):
                graph_definitions([4, 1, 2, 2, 1], (2, 2, 0, 1, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)
            if DG.isEdge(node3, node5):
                graph_definitions([4, 1, 2, 1, 2], (2, 2, 0, 1, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)
            if DG.isEdge(node4, node5):
                graph_definitions([4, 1, 1, 2, 2], (2, 2, 0, 1, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)

            if DG.isEdge(node2, node3):
                if DG.isEdge(node2, node4):
                    # graph 17
                    graph_definitions([4, 3, 2, 2, 1], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)
                    if DG.isEdge(node2, node5):
                        if DG.isEdge(node3, node4):
                            graph_definitions([4, 4, 3, 3, 2], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                            if DG.isEdge(node4, node5):
                                if DG.isEdge(node3, node5):
                                    graph_definitions([4,4,4,4,4], (0,0,0,5,0), [node1,node2,node3,node4,node5], orbital_vertex, DG)
                                graph_definitions([4, 4, 3, 4, 3], (0, 0, 2, 3, 0), [node1, node2, node3, node4, node5],orbital_vertex, DG)
                            if DG.isEdge(node3, node5):
                                graph_definitions([4, 4, 4, 3, 3], (0, 0, 2, 3, 0), [node1, node2, node3, node4, node5],orbital_vertex, DG)

                        if DG.isEdge(node4, node5):
                            graph_definitions([4, 4, 2, 3, 3], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],orbital_vertex, DG)
                            if DG.isEdge(node3, node5):
                                graph_definitions([4, 4, 3, 3, 4], (0, 0, 2, 3, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 4, 3, 2, 3], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 4, 2, 2, 2], (0, 3, 0, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node3, node4):
                        if DG.isEdge(node4, node5):
                            graph_definitions([4, 3, 3, 4, 2], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                            if DG.isEdge(node3, node5):
                                graph_definitions([4, 3, 4, 4, 3], (0, 0, 2, 3, 0), [node1, node2, node3, node4, node5],
                                                  orbital_vertex, DG)
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 3, 4, 3, 2], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)

                        graph_definitions([4, 3, 3, 3, 1], (1, 0, 3, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node4, node5):
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 3, 3, 3, 3], (0, 0, 4, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 3, 2, 3, 2], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 3, 3, 2, 2], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node2, node5):
                    if DG.isEdge(node3, node4):
                        if DG.isEdge(node4, node5):
                            if DG.isEdge(node3, node5):
                                graph_definitions([4, 3, 4, 3, 4], (0, 0, 2, 3, 0), [node1, node2, node3, node4, node5],
                                                  orbital_vertex, DG)

                            graph_definitions([4, 3, 3, 3, 3], (0, 0, 4, 1, 0), [node1, node2, node3, node4, node5],
                                                  orbital_vertex, DG)
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 3, 4, 2, 3], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)

                        graph_definitions([4, 3, 3, 2, 2], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node4, node5):
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 3, 3, 2, 4], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 3, 2, 2, 3], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 3, 3, 1, 3], (1, 0, 3, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 3, 2, 1, 2], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node3, node4):
                    if DG.isEdge(node4, node5):
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 2, 4, 3, 3], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 2, 3, 3, 2], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 2, 4, 2, 2], (0, 3, 0, 2, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 2, 3, 2, 1], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node4, node5):
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 2, 3, 2, 3], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 2, 2, 2, 2], (0, 4, 0, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node3, node5):
                    graph_definitions([4, 2, 3, 1, 2], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)

            if DG.isEdge(node2, node4):
                if DG.isEdge(node2, node5):
                    if DG.isEdge(node3, node4):
                        if DG.isEdge(node4, node5):
                            graph_definitions([4, 3, 2, 4, 3], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                            if DG.isEdge(node3, node5):
                                graph_definitions([4, 3, 3, 4, 4], (0, 0, 2, 3, 0), [node1, node2, node3, node4, node5],
                                                  orbital_vertex, DG)

                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 3, 3, 3, 3], (0, 0, 4, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 3, 2, 3, 2], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node4, node5):
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 3, 2, 3, 4], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 3, 1, 3, 3], (1, 0, 3, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 3, 2, 2, 3], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 3, 1, 2, 2], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node3, node4):
                    if DG.isEdge(node4, node5):
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 2, 3, 4, 3], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 2, 2, 4, 2], (0, 3, 0, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 2, 3, 3, 2], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 2, 2, 3, 1], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node4, node5):
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 2, 2, 3, 3], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 2, 1, 3, 2], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node3, node5):
                    graph_definitions([4, 2, 2, 2, 2], (0, 4, 0, 1, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)
            if DG.isEdge(node2, node5):
                if DG.isEdge(node3, node4):
                    if DG.isEdge(node4, node5):
                        if DG.isEdge(node3, node5):
                            graph_definitions([4, 2, 3, 3, 4], (0, 1, 2, 2, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                        graph_definitions([4, 2, 2, 3, 3], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                              orbital_vertex, DG)
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 2, 3, 2, 3], (0, 2, 2, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 2, 2, 2, 2], (0, 4, 0, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node4, node5):
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 2, 2, 2, 4], (0, 3, 0, 2, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 2, 1, 2, 3], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node3, node5):
                    graph_definitions([4, 2, 2, 1, 3], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)
            if DG.isEdge(node3, node4):
                if DG.isEdge(node4, node5):
                    if DG.isEdge(node3, node5):
                        graph_definitions([4, 1, 3, 3, 3], (1, 0, 3, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                    graph_definitions([4, 1, 2, 3, 2], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
                if DG.isEdge(node3, node5):
                    graph_definitions([4, 1, 3, 2, 2], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)

            if DG.isEdge(node4, node5):
                if DG.isEdge(node3, node5):
                    graph_definitions([4, 1, 2, 2, 3], (1, 2, 1, 1, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)
            graph_definitions([4, 1, 1, 1, 1], (4, 0, 0, 1, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)

        for (node2, node3, node4) in itertools.combinations(DG.adj_list[node1],3):  # Loop over all three-pairs of neighbors of node1
            #graph 10
            for node5 in DG.adj_list[node2]:
                if node5 != node3 and node5!= node4 and node5!= node1:
                    graph_definitions([3, 2, 1, 1, 1], (3, 1, 1, 0, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)
            for node5 in DG.adj_list[node3]:
                if node5 != node2 and node5 != node4 and node5 != node1:
                    graph_definitions([3, 1, 2, 1, 1], (3, 1, 1, 0, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)
            for node5 in DG.adj_list[node4]:
                if node5 != node3 and node5 != node2 and node5 != node1:
                    graph_definitions([3, 1, 1, 2, 1], (3, 1, 1, 0, 0), [node1, node2, node3, node4, node5],
                                  orbital_vertex, DG)

            #graph 12

            # node2 is connector and node 3 is top
            if node3 in DG.adj_list[node2]:
                for node5 in DG.adj_list[node2]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:

                        graph_definitions([3, 3, 2, 1, 1], (2, 1, 2, 0, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)

            # node3 is connector and node 2 is top
            if node3 in DG.adj_list[node2]:
                for node5 in DG.adj_list[node3]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:

                        graph_definitions([3, 2, 3, 1, 1], (2, 1, 2, 0, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)

            #node 4 top, node 2 connector
            if node4 in DG.adj_list[node2]:
                for node5 in DG.adj_list[node2]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:

                        graph_definitions([3, 3, 1, 2, 1], (2, 1, 2, 0, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)

            # node 2 top, node 4 connector
            if node4 in DG.adj_list[node2]:
                for node5 in DG.adj_list[node4]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:

                        graph_definitions([3, 2, 1, 3, 1], (2, 1, 2, 0, 0), [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)

            #node 3 connector and node 4 as top
            if node4 in DG.adj_list[node3]:
                for node5 in DG.adj_list[node3]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5!= node4:

                        graph_definitions([3, 1, 3, 2, 1], (2, 1, 2, 0, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)

            # node 4 connector and node 3 as top
            if node4 in DG.adj_list[node3]:
                for node5 in DG.adj_list[node4]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:
                        graph_definitions([3, 1, 2, 3, 1], (2, 1, 2, 0, 0),
                                          [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)

            #graph 13
            #for usability 4 designates the deg 2 connected to single vertex
            if node2 in DG.adj_list[node3]:
                for node5 in DG.adj_list[node4]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:
                        graph_definitions([3, 2, 2, 4, 1], (1, 3, 1, 0, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)
            if node3 in DG.adj_list[node4]:
                for node5 in DG.adj_list[node2]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:
                        graph_definitions([3, 4, 2, 2, 1], (1, 3, 1, 0, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)
            if node4 in DG.adj_list[node2]:
                for node5 in DG.adj_list[node3]:
                    if node5 != node1 and node5 != node3 and node5 != node2 and node5 != node4:
                        graph_definitions([3, 2, 4, 2, 1], (1, 3, 1, 0, 0), [node1, node2, node3, node4, node5],
                                      orbital_vertex, DG)

            #graph 16
            for node5a in DG.adj_list[node2]:
                if node5a != node1 and node5a != node3 and node5a != node2 and node5a != node4:
                    for node5b in DG.adj_list[node3]:
                        if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                            if node5a == node5b:
                                graph_definitions([3, 2, 2, 1, 5], (1, 3, 1, 0, 0), [node1, node2, node3, node4, node5a],
                                              orbital_vertex, DG)
                    for node5c in DG.adj_list[node4]:
                        if node5c != node1 and node5c != node3 and node5c != node2 and node5c != node4:
                            if node5a == node5c:
                                graph_definitions([3, 2, 1, 2, 5], (1, 3, 1, 0, 0), [node1, node2, node3, node4, node5c],
                                                  orbital_vertex, DG)
            for node5d in DG.adj_list[node3]:
                if node5d != node1 and node5d != node3 and node5d != node2 and node5d != node4:
                    for node5e in DG.adj_list[node4]:
                        if node5e != node1 and node5e != node3 and node5e != node2 and node5e != node4:
                            if node5e == node5d:
                                graph_definitions([3, 1, 2, 2, 5], (1, 3, 1, 0, 0), [node1, node2, node3, node4, node5e],
                                          orbital_vertex, DG)

            #graph 19
            for node5a in DG.adj_list[node2]:
                if node5a != node1 and node5a != node3 and node5a != node2 and node5a != node4:
                    for node5b in DG.adj_list[node3]:
                        if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                            if node5a == node5b:
                                if node2 in DG.adj_list[node3]:
                                    graph_definitions([3, 3, 3, 1, 2], (1, 1, 3, 0, 0),[node1, node2, node3, node4, node5a],
                                                      orbital_vertex, DG)
                    for node5c in DG.adj_list[node4]:
                        if node5c != node1 and node5c != node3 and node5c != node2 and node5c != node4:
                            if node5a == node5c:
                                if node2 in DG.adj_list[node4]:
                                    graph_definitions([3, 3, 1, 3, 2], (1, 1, 3, 0, 0),[node1, node2, node3, node4, node5c],
                                                      orbital_vertex, DG)
            for node5d in DG.adj_list[node3]:
                for node5e in DG.adj_list[node4]:
                    if node5e==node5d:
                        if node5e != node1 and node5e != node3 and node5e != node2 and node5e != node4:
                            if node3 in DG.adj_list[node4]:
                                graph_definitions([3, 1, 3, 3, 2], (1, 1, 3, 0, 0), [node1, node2, node3, node4, node5e],
                                                  orbital_vertex, DG)

            #graph 20
            for node5a in DG.adj_list[node2]:
                for node5b in DG.adj_list[node3]:
                    for node5c in DG.adj_list[node4]:
                        if node5a == node5b and node5b == node5c:
                            if node5a != node1 and node5a != node3 and node5a != node2 and node5a != node4:
                                #print('hello')
                                # deg 4 actually == deg 3 and deg5 actually == deg2
                                graph_definitions([4, 5, 5, 5, 4], (0, 3, 2, 0, 0), [node1, node2, node3, node4, node5a],
                                              orbital_vertex, DG)

            #graph 21
            #for all graph 21 orbit 52 is deg2 but designated at deg 4
            if node2 in DG.adj_list[node3]:
                for node5a in DG.adj_list[node4]:
                    for node5b in DG.adj_list[node3]:
                        if node5a != node2 and node5a == node5b:
                            if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                graph_definitions([3, 4, 3, 2, 2], (0, 3, 2, 0, 0), [node1, node2, node3, node4, node5b],
                                              orbital_vertex, DG)
            if node3 in DG.adj_list[node4]:
                for node5a in DG.adj_list[node4]:
                    for node5b in DG.adj_list[node2]:
                        if node5a != node3 and node5a == node5b:
                            if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                graph_definitions([3, 2, 4, 3, 2], (0, 3, 2, 0, 0), [node1, node2, node3, node4, node5b],
                                              orbital_vertex, DG)
            if node4 in DG.adj_list[node3]:
                for node5a in DG.adj_list[node3]:
                    for node5b in DG.adj_list[node2]:
                        if node5a != node4 and node5a == node5b:
                            if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                graph_definitions([3, 2, 3, 4, 2], (0, 3, 2, 0, 0),
                                              [node1, node2, node3, node4, node5b],
                                              orbital_vertex, DG)
            if node2 in DG.adj_list[node4]:
                for node5a in DG.adj_list[node3]:
                    for node5b in DG.adj_list[node4]:
                        if node5a != node4 and node5a == node5b:
                            if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                graph_definitions([3, 4, 2, 3, 2], (0, 3, 2, 0, 0),
                                              [node1, node2, node3, node4, node5b],
                                              orbital_vertex, DG)
            if node3 in DG.adj_list[node2]:
                for node5a in DG.adj_list[node2]:
                    for node5b in DG.adj_list[node4]:
                        if node5a != node4 and node5a == node5b:
                            if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                graph_definitions([3, 3, 4, 2, 2], (0, 3, 2, 0, 0),
                                              [node1, node2, node3, node4, node5b],
                                              orbital_vertex, DG)
            if node4 in DG.adj_list[node2]:
                for node5a in DG.adj_list[node2]:
                    for node5b in DG.adj_list[node3]:
                        if node5a != node4 and node5a == node5b:
                            if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                graph_definitions([3, 3, 2, 4, 2], (0, 3, 2, 0, 0),
                                              [node1, node2, node3, node4, node5b],
                                              orbital_vertex, DG)

            #graph 25
            #for all graph 25, orbit 64 with deg 2 will be deg 4
            if node2 in DG.adj_list[node3]:
                if node2 in DG.adj_list[node4]:
                    for node5a in DG.adj_list[node3]:
                        for node5b in DG.adj_list[node4]:
                            if node5a == node5b and node5b != node2:
                                if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                    graph_definitions([3, 3, 4, 4, 2], (0, 1, 4, 0, 0),
                                                  [node1, node2, node3, node4, node5b],
                                                  orbital_vertex, DG)
            if node4 in DG.adj_list[node3]:
                if node4 in DG.adj_list[node2]:
                    for node5a in DG.adj_list[node3]:
                        for node5b in DG.adj_list[node2]:
                            if node5a == node5b and node5b != node4:
                                if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                    graph_definitions([3, 4, 4, 3, 2], (0, 1, 4, 0, 0),
                                                  [node1, node2, node3, node4, node5b],
                                                  orbital_vertex, DG)
            if node3 in DG.adj_list[node4]:
                if node3 in DG.adj_list[node2]:
                    for node5a in DG.adj_list[node4]:
                        for node5b in DG.adj_list[node2]:
                            if node5a == node5b and node5b != node3:
                                if node5b != node1 and node5b != node3 and node5b != node2 and node5b != node4:
                                    graph_definitions([3, 4, 3, 4, 2], (0, 1, 4, 0, 0),
                                                  [node1, node2, node3, node4, node5b],
                                                  orbital_vertex, DG)


        for (node2, node3) in itertools.combinations(DG.adj_list[node1],2):
            #graph 9
            for node4 in DG.adj_list[node2]:
                for node5 in DG.adj_list[node3]:
                    if node5 != node4 and node5 != node2 and node4 != node3 and node5 != node1 and node4 != node1:
                        graph_definitions([2, 2, 2, 1, 1], (2, 3, 0, 0, 0),
                                          [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
            #graph 15
            for node4 in DG.adj_list[node2]:
                for node5 in DG.adj_list[node3]:
                    if node5 != node4 and node4 != node1 and node5 != node1 and node5!= node2 and node4 != node3 and node5 in DG.adj_list[node4]:
                        #print(node1, node2, node3, node4, node5)
                        graph_definitions([2, 2, 2, 2, 2], (0, 5, 0, 0, 0),
                                          [node1, node2, node3, node4, node5],
                                          orbital_vertex, DG)
    return orbital_vertex


def four_path_based(G,fname='',gname='',to_list=False):
    
    tailed_cycle = 0
    hatted_cycle = 0

    tailed_clique = 0
    hatted_clique = 0
    four_paths = 0
    double_tailed_tri = 0
    long_tailed_tri = 0
    fork_tailed_tri = 0
    hourglass = 0
    cobra = 0
    stingray = 0
    fivecycle = 0
    triangle_strip = 0

    for node1 in G.vertices:
        for node2 in G.adj_list[node1]:
            for node3 in G.adj_list[node2]:
                if node3 == node1:
                    continue
                for node4 in G.adj_list[node3]:
                    if node4 == node2 or node4 == node1:
                        continue;
                    for node5 in G.adj_list[node4]:
                        if node5 == node3 or node5 == node1:
                            continue
                        if node5 == node2:
                            for node6 in G.adj_list[node2]:
                                if node6 == node1 or node6 == node3 or node6 == node4:
                                    continue
                                fork_tailed_tri = fork_tailed_tri+1
                            continue
                        four_paths = four_paths+1
                        
                        if G.isEdge(node2,node4):
                            double_tailed_tri = double_tailed_tri+1

                        if G.isEdge(node3,node5):
                            long_tailed_tri = long_tailed_tri+1

                        if G.isEdge(node1,node3) and G.isEdge(node5,node3):
                            hourglass = hourglass + 1

                        if G.isEdge(node2,node5) and G.isEdge(node3,node5):
                            cobra = cobra + 1

                        if G.isEdge(node2,node5) and G.isEdge(node2,node4):
                            stingray = stingray+1
                        
                        if G.isEdge(node5,node1):
                            fivecycle = fivecycle+1
                        
                        if G.isEdge(node1,node5) and G.isEdge(node2,node5) and G.isEdge(node2,node4):
                            triangle_strip = triangle_strip+1
                        
                        if G.isEdge(node2,node5):
                            tailed_cycle = tailed_cycle+1
                            if to_list:
                                print("Tailed cycle:",node1,node2,node3,node4,node5)
                            
                            if G.isEdge(node3,node5) and G.isEdge(node2,node4):
                                tailed_clique = tailed_clique+1
                                if to_list:
                                    print("Tailed clique:",node1,node2,node3,node4,node5)

                            if G.isEdge(node1,node3):
                                hatted_cycle = hatted_cycle+1
                                if to_list:
                                    print("Hatted cycle:",node1,node2,node3,node4,node5)

                                if G.isEdge(node3,node5) and G.isEdge(node2,node4):
                                    hatted_clique = hatted_clique+1
                                    if to_list:
                                        print("Hatted clique:",node1,node2,node3,node4,node5)

    return (tailed_cycle/2,hatted_cycle/2,tailed_clique/6,hatted_clique/4,four_paths/2,double_tailed_tri/2,long_tailed_tri/2,fork_tailed_tri/4,hourglass/8,cobra/2,stingray/2,fivecycle/10,triangle_strip/2)


