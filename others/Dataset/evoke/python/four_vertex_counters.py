import itertools
import sys
import triangle_counters
import numpy as np

##### This file contains functions to count all 4-vertex patterns in a graph.
#####
##### C. Seshadhri, Feb 2015

### 4_vertex_count(G) counts all 4-vertex patterns in G. It uses different methods
### for each pattern, which are all explained in the functions. The output
### is a list of 6 numbers, corresponding to the *induced* counts of:
### 3-stars  3-paths  tailed-triangles   4-cycles   chordal-4-cycles   4-cliques
#### the graph is a DAG, this is exactly the number of triangles.
####
#### Optional arguments:
####
#### fname, gname: If these are supplied, extra information about the work done by
#### the 4_vertex_count is dumped in fname. The graph name is given by gname.
####

def four_vertex_count(G,fname='',gname='',want_induced=True):
    order = G.DegenOrdering()   # Get degeneracy ordering
    DG = G.Orient(order)        # DG is digraph with this orientation
    print('Got degeneracy orientation')
#     DG = G.DegreeOrder()

    size = G.Size()
    tri_info = triangle_counters.triangle_info(DG) # Get triangle info

    #easier readability
    tri_vertex = tri_info[0]
    tri_edge = tri_info[1]
    print('Got triangle information')

    triangle = sum(tri_vertex.values())/3     # Sum of per-vertex triangle counts is 3 times the total triangle count

    if fname != '':             # if fname is actually given
        f_out = open(fname,'a')
        f_out.write(gname+':\n')
        f_out.write('n = '+str('%0.2E'%size[0])+'    m = '+str('%0.2E'%(size[1]/2))+'    W = '+str('%0.2E'%size[2])+'    T = '+str('%0.2E'%triangle)+'\n')

    star_3 = 0.0
    path_3 = 0.0
    tailed_triangle = 0.0
    cycle_4 = 0.0
    chordal_cycle = 0.0
    clique_4 = 0.0

    debug = 0

    for node in G.vertices:
        deg = G.degrees[node]
        tri = tri_info[0][node]
        star_3 += deg*(deg-1)*(deg-2)/6    # Number of 3-stars = \sum_v {d_v \choose 3}
        tailed_triangle += (deg-2)*tri      # Number of tailed triangles hinged at v = (d_v-2)*t_v

        for nbr in G.adj_list[node]:             # Loop over neighbors of node
            deg_nbr = G.degrees[nbr]
            path_3 += (deg-1)*(deg_nbr-1)  # Number of 3-paths involving edge (node,nbr) = (deg-1)(deg_nbr-1)
            tri_edge = tri_info[1][(node,nbr)]
            chordal_cycle += tri_edge*(tri_edge-1)/2 # Number of chordal-cycles hinged at edge e = {d_e \choose 2}

    # Previous code counts each 3-path twice because each edge appears twice in loop. After this correction
    # each triangle is counted thrice as a 3-path
    path_3 = path_3/2 - 3*triangle

    # Previous code counts each chordal-cycle twice because each edge appears twice in loop.
    chordal_cycle = chordal_cycle/2
    print('Computed everything but 4-cycles and 4-cliques')

    wedge_outout = {}       # Hash tables for storing wedges

    # The directed interpretation of Chiba-Nishizeki: for each (u,v), count the number of out wedges and in-out wedges with ends (u,v)

    # There are 3-types of directed 4-cycles
    type1 = 0.0
    type2 = 0.0
    type3 = 0.0

    outout_nz = 0.0
    inout_nz = 0.0

    outout = 0.0
    inout = 0.0

    for node in DG.vertices:
        # First we index out-out wedges centered at node
        for (nbr1, nbr2) in itertools.combinations(DG.adj_list[node],2):    #Loop over all pairs of neighbors of node1
            if nbr1 > nbr2:     # If nbr1 > nbr2, swap, so that nbr1 \leq nbr2
                tmp = nbr1
                nbr1 = nbr2
                nbr2 = tmp

           # print(node,nbr1,nbr2)

            if (nbr1,nbr2) in wedge_outout:    # If (nbr1,nbr2) already seen, increment wedge count
                wedge_outout[(nbr1,nbr2)] += 1
                outout += 1
            else:
                outout_nz += 1
                wedge_outout[(nbr1,nbr2)] = 1  # Else initialize wedge count to 1

    print('Out-out pairs = ',outout_nz)

    for node in DG.vertices:
        endpoints = {}
        for nbr1 in DG.adj_list[node]:
            for nbr2 in DG.adj_list[nbr1]:       # Get in-out wedge with source at node
                if nbr2 in endpoints:
                    endpoints[nbr2] += 1
                    inout += 1
                else:
                    endpoints[nbr2] = 1
                    inout_nz += 1

        for v in endpoints:
            count = endpoints[v]
            type2 += count*(count-1)/2

            v1 = node
            v2 = v

            if v1 > v2:
                swp = v1
                v1 = v2
                v2 = swp

            if (v1,v2) in wedge_outout:
                type3 += count*wedge_outout[(v1,v2)]

    print('In-out pairs =',inout_nz)

    if debug:
        print(wedge_outout)

    for pair in wedge_outout:       # Loop over all pairs in wedge_outout
        outout += 1
        count = wedge_outout[pair]
        type1 += count*(count-1)/2  # Number of type1 4-cycles hinged at (u,v) = {W^{++}_{u,v} \choose 2}

    cycle_4 = type1 + type2 + type3

    print('Computed 4-cycle count')
    print('type1 = ',type1,', type2 = ',type2,', type3 = ',type3)


    clique_work = 0.0
    for node in DG.vertices:        # Loop over nodes
        nbrs = DG.adj_list[node]
        nbrs_info = []
        for cand in nbrs:           # Get topological order position for each cand in nbrs
            nbrs_info.append((cand,DG.top_order_inv[cand]))

        sorted_nbrs = sorted(nbrs_info, key=lambda entry: entry[1])   # Sort nbrs according to position in topological ordering

        deg = len(sorted_nbrs)      # Out-degree of node
        for i in range(0,deg):      # Loop over neighbors in sorted order
            nbri = sorted_nbrs[i][0]

            # Get all vertices nbrj > nbri that form triangle with nbri
            tri_end = []
            for j in range(i+1,deg):   # Loop over tuple of neighbors i < j
                nbrj = sorted_nbrs[j][0]
                if G.isEdge(nbri,nbrj):
                    tri_end.append(nbrj)  # nbrj forms triangle with (node,nbri)

            # Now look for edges among pairs in tri_end, to find 4-cliques
            for (v1, v2) in itertools.combinations(tri_end,2):
                clique_work += 1
                if G.isEdge(v1,v2):
                    clique_4 += 1

    print('Got cliques. Searched over',clique_work,'tuples')

    transform = np.matrix('1 0 1 0 2 4; 0 1 2 4 6 12; 0 0 1 0 4 12; 0 0 0 1 1 3; 0 0 0 0 1 6; 0 0 0 0 0 1')

    non_induced_counts = [star_3, path_3, tailed_triangle, cycle_4, chordal_cycle, clique_4]

    print(non_induced_counts)

    induced =  np.linalg.solve(transform,non_induced_counts)

    if fname != '':
        f_out.write('out-out = '+str('%0.2E'%outout)+'    out-out-nz = '+str('%0.2E'%outout_nz)+'\n')
        f_out.write('in-out = '+str('%0.2E'%inout)+'    in-out-nz = '+str('%0.2E'%inout_nz)+'\n')
        success_rate = clique_4/clique_work
        f_out.write('clique_work = '+str('%0.2E'%clique_work)+'    success rate = '+str('%0.2f'%success_rate)+'\n')
        induced_str = ''
        for i in range(0,6):
            induced_str = induced_str + str('%0.2E'%induced[i])+'   '
        f_out.write(induced_str+'\n\n\n')
        f_out.close()

    if want_induced:
        return induced
    return non_induced_counts


#### This function runs the 4-clique counting heuristic based on degeneracy ordering, and stores
#### the sizes and densities of every egonet processed. The output is dumped in fname, where
#### each line has the size in vertices and edge density of every egonet.
####

def four_clique_data(G,fname):

    order = G.DegenOrdering()   # Get degeneracy ordering
    DG = G.Orient(order)        # DG is digraph with this orientation
    print('Got degeneracy orientation')

    f_out = open(fname,'w')

    for node in DG.vertices:        # Loop over nodes
        nbrs = DG.adj_list[node]
        nbrs_info = []
        for cand in nbrs:           # Get topological order position for each cand in nbrs
            nbrs_info.append((cand,DG.top_order_inv[cand]))

        sorted_nbrs = sorted(nbrs_info, key=lambda entry: entry[1])   # Sort nbrs according to position in topological ordering

        deg = len(sorted_nbrs)      # Out-degree of node
        for i in range(0,deg):      # Loop over neighbors in sorted order
            nbri = sorted_nbrs[i][0]

            # Get all vertices nbrj > nbri that form triangle with nbri
            tri_end = []
            for j in range(i+1,deg):   # Loop over tuple of neighbors i < j
                nbrj = sorted_nbrs[j][0]
                if G.isEdge(nbri,nbrj):
                    tri_end.append(nbrj)  # nbrj forms triangle with (node,nbri)

            # Now look for edges among pairs in tri_end, to find 4-cliques
            clique_4 = 0.0
            clique_work = 0.0
            for (v1, v2) in itertools.combinations(tri_end,2):
                clique_work += 1
                if G.isEdge(v1,v2):
                    clique_4 += 1

            if clique_work > 0:
                f_out.write(str(len(tri_end))+'  '+str('%0.2f'%(clique_4/clique_work))+'\n')

    f_out.close()


def four_vertex_orbital(G, orbital_vertex,fname='',gname=''):
    # order = G.DegenOrdering()   # Get degeneracy ordering
    # DG = G.Orient(order)        # DG is digraph with this orientation
    # print('Got degeneracy orientation')
#     DG = G.DegreeOrder()
    DG = G
    size = G.Size()
    for node1 in DG.vertices:     # Loop over all nodes
        for (node2, node3, node4) in itertools.combinations(DG.adj_list[node1],3):    #Loop over all three-pairs of neighbors of node1
            if DG.isEdge(node2, node3):
                if DG.isEdge(node2, node4):
                    if DG.isEdge(node3, node4):
                        #graph G8
                        orbital_vertex[node1][14]+=1
                        orbital_vertex[node2][14]+=1
                        orbital_vertex[node3][14]+=1
                        orbital_vertex[node4][14]+=1
                    #graph G7
                    orbital_vertex[node2][13]+=1
                    orbital_vertex[node1][13]+=1
                    orbital_vertex[node3][12]+=1
                    orbital_vertex[node4][12]+=1
                if DG.isEdge(node3, node4):
                    #graph G7
                    orbital_vertex[node3][13] += 1
                    orbital_vertex[node1][13] += 1
                    orbital_vertex[node2][12] += 1
                    orbital_vertex[node4][12] += 1

                #graph G6
                orbital_vertex[node3][10] += 1
                orbital_vertex[node1][11] += 1
                orbital_vertex[node2][10] += 1
                orbital_vertex[node4][9] += 1

            if DG.isEdge(node3, node4):
                if DG.isEdge(node2, node4):
                    # graph G7
                    orbital_vertex[node4][13]+=1
                    orbital_vertex[node1][13]+=1
                    orbital_vertex[node3][12]+=1
                    orbital_vertex[node2][12]+=1

                #graph G6
                orbital_vertex[node3][10] += 1
                orbital_vertex[node1][11] += 1
                orbital_vertex[node4][10] += 1
                orbital_vertex[node2][9] += 1

            if DG.isEdge(node2, node4):
                #graph G6
                orbital_vertex[node2][10] += 1
                orbital_vertex[node1][11] += 1
                orbital_vertex[node4][10] += 1
                orbital_vertex[node3][9] += 1

            #graph G4
            orbital_vertex[node2][6] += 1
            orbital_vertex[node1][7] += 1
            orbital_vertex[node4][6] += 1
            orbital_vertex[node3][6] += 1

    return orbital_vertex

def four_vertex_edge(G, orbital_edge, fname='', gname=''):
    order = G.DegenOrdering()  # Get degeneracy ordering
    DG = G.Orient(order)  # DG is digraph with this orientation
    print('Got degeneracy orientation')
    #     DG = G.DegreeOrder()
    size = G.Size()
    for node1 in DG.vertices:  # Loop over all nodes
        for (node2, node3, node4) in itertools.combinations(DG.adj_list[node1],
                                                            3):  # Loop over all three-pairs of neighbors of node1
            if DG.isEdge(node2, node3):
                if DG.isEdge(node2, node4):
                    if DG.isEdge(node3, node4):
                        # graph G8
                        orbital_edge[(node1, node3)][11] += 1
                        orbital_edge[(node1, node2)][11] += 1
                        orbital_edge[(node1, node4)][11] += 1
                        orbital_edge[(node2, node1)][11] += 1
                        orbital_edge[(node3, node1)][11] += 1
                        orbital_edge[(node4, node1)][11] += 1
                        orbital_edge[(node2, node4)][11] += 1
                        orbital_edge[(node2, node3)][11] += 1
                        orbital_edge[(node4, node2)][11] += 1
                        orbital_edge[(node3, node2)][11] += 1
                        orbital_edge[(node3, node4)][11] += 1
                        orbital_edge[(node4, node3)][11] += 1
                    else:
                        # graph G7
                        orbital_edge[(node1, node3)][9] += 1
                        orbital_edge[(node1, node2)][10] += 1
                        orbital_edge[(node1, node4)][9] += 1
                        orbital_edge[(node2, node1)][10] += 1
                        orbital_edge[(node3, node1)][9] += 1
                        orbital_edge[(node4, node1)][9] += 1
                        orbital_edge[(node2, node4)][9] += 1
                        orbital_edge[(node2, node3)][9] += 1
                        orbital_edge[(node4, node2)][9] += 1
                        orbital_edge[(node3, node2)][9] += 1

                elif DG.isEdge(node3, node4):
                    # graph G7
                    orbital_edge[(node1, node3)][10] += 1
                    orbital_edge[(node1, node2)][9] += 1
                    orbital_edge[(node1, node4)][9] += 1
                    orbital_edge[(node2, node1)][9] += 1
                    orbital_edge[(node3, node1)][10] += 1
                    orbital_edge[(node4, node1)][9] += 1

                    orbital_edge[(node2, node3)][9] += 1

                    orbital_edge[(node3, node2)][9] += 1
                    orbital_edge[(node3, node4)][9] += 1
                    orbital_edge[(node4, node3)][9] += 1
                else:
                    # graph G6
                    orbital_edge[(node1, node3)][8] += 1
                    orbital_edge[(node1, node2)][8] += 1
                    orbital_edge[(node1, node4)][6] += 1
                    orbital_edge[(node2, node1)][8] += 1
                    orbital_edge[(node3, node1)][8] += 1
                    orbital_edge[(node4, node1)][6] += 1
                    orbital_edge[(node2, node3)][7] += 1
                    orbital_edge[(node3, node2)][7] += 1

            elif DG.isEdge(node3, node4):
                if DG.isEdge(node2, node4):
                    # graph G7
                    orbital_edge[(node1, node3)][9] += 1
                    orbital_edge[(node1, node2)][9] += 1
                    orbital_edge[(node1, node4)][10] += 1
                    orbital_edge[(node2, node1)][9] += 1
                    orbital_edge[(node3, node1)][9] += 1
                    orbital_edge[(node4, node1)][10] += 1
                    orbital_edge[(node2, node4)][9] += 1

                    orbital_edge[(node4, node2)][9] += 1

                    orbital_edge[(node3, node4)][9] += 1
                    orbital_edge[(node4, node3)][9] += 1
                else:
                    # graph G6
                    orbital_edge[(node1, node3)][8] += 1
                    orbital_edge[(node1, node2)][6] += 1
                    orbital_edge[(node1, node4)][8] += 1
                    orbital_edge[(node2, node1)][6] += 1
                    orbital_edge[(node3, node1)][8] += 1
                    orbital_edge[(node4, node1)][8] += 1
                    orbital_edge[(node3, node4)][7] += 1
                    orbital_edge[(node4, node3)][7] += 1
            elif DG.isEdge(node2, node4):
                # graph G6
                orbital_edge[(node1, node3)][6] += 1
                orbital_edge[(node1, node2)][8] += 1
                orbital_edge[(node1, node4)][8] += 1
                orbital_edge[(node2, node1)][8] += 1
                orbital_edge[(node3, node1)][6] += 1
                orbital_edge[(node4, node1)][8] += 1
                orbital_edge[(node2, node4)][7] += 1
                orbital_edge[(node4, node2)][7] += 1

            else:
                # graph G4
                orbital_edge[(node1, node3)][4] += 1
                orbital_edge[(node1, node2)][4] += 1
                orbital_edge[(node1, node4)][4] += 1
                orbital_edge[(node2, node1)][4] += 1
                orbital_edge[(node3, node1)][4] += 1
                orbital_edge[(node4, node1)][4] += 1
    return orbital_edge