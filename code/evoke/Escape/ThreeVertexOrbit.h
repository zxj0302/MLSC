#ifndef ESCAPE_THREEVERTEXEDGEORBIT_H_
#define ESCAPE_THREEVERTEXEDGEORBIT_H_

#include <algorithm>

#include "Escape/ErrorCode.h"
#include "Escape/Graph.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/Utils.h"
#include <iostream>

using namespace std;
using namespace Escape;

// This is a function that counts edge appearances in edge orbits of 3-vertex patterns
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG) and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have all the edge orbit counts for 3-vertex patterns

void three_edge_orbit_counter(CGraph *gout, OrbitInfo &orbit_counts)
{
  // OLD OLD OLd
  // VertexIdx j, k;
  // EdgeIdx loc;
  // OLD OLD OLD
  
   // OLD OLD OLD
   // for(EdgeIdx i = 0; i < gout->nEdges; i++)
   // {
   //  orbit_counts.per_edge_[0][i] = 0;
   //  orbit_counts.per_edge_rev_[0][i] = 0;
   //  orbit_counts.per_edge_[1][i] = 0;
   //  orbit_counts.per_edge_rev_[1][i] = 0;
   // }
   // OLD OLD OLD

   for (VertexIdx i=0; i < gout->nVertices; ++i) // loop over vertices
   {
       for (EdgeIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; posj++)   // loop over neighbor of i
       {
            VertexIdx j = gout->nbors[posj];
            VertexIdx deg_i = orbit_counts.per_vertex_[0][i];
            VertexIdx deg_j = orbit_counts.per_vertex_[0][j];
            orbit_counts.per_edge_[0][posj] += deg_i-1;
            orbit_counts.per_edge_rev_[0][posj] += deg_j-1; 

            for (EdgeIdx posk = posj+1; posk < gout->offsets[i+1]; posk++)         // loop over another neighbor of i
            {
               VertexIdx k = gout->nbors[posk];
               EdgeIdx loc = gout->getEdgeBinary(j, k);
               if (loc != -1)  // (j, k) is present
               {
                   // update all per edge counts. Note that location used is same as position in g->nbors
                   orbit_counts.per_edge_[1][posj]++;
                   orbit_counts.per_edge_[1][posk]++;
                   orbit_counts.per_edge_[1][loc]++;
               }
            }
       }
   }

}


#endif



