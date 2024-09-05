#ifndef ESCAPE_FOURVERTEXORBIT_H_
#define ESCAPE_FOURVERTEXORBIT_H_

#include <iostream>
#include <algorithm>

#include "Escape/ErrorCode.h"
#include "Escape/Graph.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/Utils.h"


using namespace std;
using namespace Escape;

// This is a function that counts Node appearances in node orbit 8 (the only orbit in a 4-cylce)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbit 8 (the only orbit in a 4-cylce)

void four_cycle_orbit_counter_node_only(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{

   VertexIdx i,j,k;
   EdgeIdx pos_ji, pos_kj;

   VertexIdx *wedge_count = new VertexIdx[gout->nVertices+1]; //stores number of wedges ending at a vertex

   for (i=0; i < gout->nVertices; i++) //initialize all wedge_count values to 0
       wedge_count[i] = 0;

   for (i=0; i < gin->nVertices; ++i) // loop over vertices
   {
       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k >= i)   // break ties to prevent overcount
                   continue;
               wedge_count[k]++; // increment number of wedges ending at k
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               wedge_count[k]++; // increment number of wedges ending at k
           }
       }

       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           pos_ji = gout->getEdgeBinary(j,i); // pos_ji is the index of edge j -> i (from gout)

           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k >= i)   // break ties to prevent overcount
                   continue;

               // every pair of wedges ending at k yields a four-cycle
               orbit_counts.per_vertex_[8][j] += max((Count) 0, wedge_count[k]-1);
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               pos_kj = gout->getEdgeBinary(k,j);
               
               // every pair of wedges ending at k yields a four-cycle
               orbit_counts.per_vertex_[8][j] += max((Count) 0, wedge_count[k]-1);
           }
       }

       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k >= i)   // break ties to prevent overcount
                   continue;

               orbit_counts.per_vertex_[8][i] += choosec(wedge_count[k], 2);
               orbit_counts.per_vertex_[8][k] += choosec(wedge_count[k], 2);
               wedge_count[k] = 0; //reset value of wedge_count
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j

               orbit_counts.per_vertex_[8][i] += choosec(wedge_count[k], 2);
               orbit_counts.per_vertex_[8][k] += choosec(wedge_count[k], 2);
               wedge_count[k] = 0; //reset value of wedge_count
           }
       }

   }
   delete [] wedge_count;
}



// This is a function that counts Node appearances in node orbit 8 (the only orbit in a 4-cylce)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbit 8 (the only orbit in a 4-cylce)

void four_cycle_orbit_counter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{

   VertexIdx i,j,k;
   EdgeIdx pos_ji, pos_kj;

   VertexIdx *wedge_count = new VertexIdx[gout->nVertices+1]; //stores number of wedges ending at a vertex

   for (i=0; i < gout->nVertices; i++) //initialize all wedge_count values to 0
       wedge_count[i] = 0;

   for (i=0; i < gin->nVertices; ++i) // loop over vertices
   {
       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k >= i)   // break ties to prevent overcount
                   continue;
               wedge_count[k]++; // increment number of wedges ending at k
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               wedge_count[k]++; // increment number of wedges ending at k
           }
       }

       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           pos_ji = gout->getEdgeBinary(j,i); // pos_ji is the index of edge j -> i (from gout)

           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k >= i)   // break ties to prevent overcount
                   continue;

               // every pair of wedges ending at k yields a four-cycle
               orbit_counts.per_vertex_[8][j] += max((Count) 0, wedge_count[k]-1);
               orbit_counts.per_edge_[5][pos_ji] += max((Count) 0, wedge_count[k]-1);
               orbit_counts.per_edge_[5][next] += max((Count) 0, wedge_count[k]-1); // next is the index of the edge j -> k (in gout)

           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               pos_kj = gout->getEdgeBinary(k,j);
               
               // every pair of wedges ending at k yields a four-cycle
               orbit_counts.per_vertex_[8][j] += max((Count) 0, wedge_count[k]-1);
               orbit_counts.per_edge_[5][pos_ji] += max((Count) 0, wedge_count[k]-1);
               orbit_counts.per_edge_[5][pos_kj] += max((Count) 0, wedge_count[k]-1);
           }
       }

       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k >= i)   // break ties to prevent overcount
                   continue;

               orbit_counts.per_vertex_[8][i] += choosec(wedge_count[k], 2);
               orbit_counts.per_vertex_[8][k] += choosec(wedge_count[k], 2);
               wedge_count[k] = 0; //reset value of wedge_count
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j

               orbit_counts.per_vertex_[8][i] += choosec(wedge_count[k], 2);
               orbit_counts.per_vertex_[8][k] += choosec(wedge_count[k], 2);
               wedge_count[k] = 0; //reset value of wedge_count
           }
       }

   }
   delete [] wedge_count;
}

// This is a function that counts Node appearances in orbits 9, 10, 12, and 13
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gout to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts

void otherFourPatternOrbitCounterNodeOnly(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    VertexIdx i, j, k;
    EdgeIdx loc;
    for (i=0; i < gout->nVertices; ++i) // loop over vertices
    {
       for (EdgeIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; posj++)   // loop over out-neighbor of i
       {
          j = gout->nbors[posj];

          orbit_counts.per_vertex_[9][i] += orbit_counts.per_vertex_[3][j] - orbit_counts.per_edge_[1][posj];
          orbit_counts.per_vertex_[10][i] += orbit_counts.per_edge_[1][posj] * (orbit_counts.per_vertex_[0][j] - 2);
          orbit_counts.per_vertex_[13][j] += choosec(orbit_counts.per_edge_[1][posj], 2);
          
          for (VertexIdx posk = posj+1; posk < gout->offsets[i+1]; posk++) // loop over another out-neighbor of i, that is "ahead" of j in list of out-neighbors
          {
                k = gout->nbors[posk];
                loc = gout->getEdgeBinary(j,k); 
                if(loc != -1) // (j, k) is present
                {
                    orbit_counts.per_vertex_[12][i] += orbit_counts.per_edge_[1][loc] - 1;
                    orbit_counts.per_vertex_[12][j] += orbit_counts.per_edge_[1][posk] - 1;
                    orbit_counts.per_vertex_[12][k] += orbit_counts.per_edge_[1][posj] - 1;
                }
          }
        }

       for (EdgeIdx posj = gin->offsets[i]; posj < gin->offsets[i+1]; posj++)   // loop over in-neighbor of i
       {
          j = gin->nbors[posj];
          loc = gout->getEdgeBinary(j, i);
          orbit_counts.per_vertex_[9][i] += orbit_counts.per_vertex_[3][j] - orbit_counts.per_edge_[1][loc];
          orbit_counts.per_vertex_[10][i] += orbit_counts.per_edge_[1][loc] * (orbit_counts.per_vertex_[0][j] - 2);
          orbit_counts.per_vertex_[13][j] += (orbit_counts.per_edge_[1][loc] * (orbit_counts.per_edge_[1][loc] - 1)) / 2;
        }
    }
}


// This is a function that counts Node appearances in orbits 9, 10, 12, and 13, and also edge appearances in edge orbits 2, 3, 4, 6, 7, 8, and 9
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gout to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts

void otherFourPatternOrbitCounter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    VertexIdx i, j, k;
    EdgeIdx loc;
    for (i=0; i < gout->nVertices; ++i) // loop over vertices
    {
       for (EdgeIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; posj++)   // loop over out-neighbor of i
       {
          j = gout->nbors[posj];
          
          orbit_counts.per_edge_[3][posj] += (orbit_counts.per_vertex_[0][i] - 1) * (orbit_counts.per_vertex_[0][j] - 1) - orbit_counts.per_edge_[1][posj];
          
          orbit_counts.per_edge_[4][posj] += choosec(orbit_counts.per_vertex_[0][i] - 1, 2); 
          orbit_counts.per_edge_rev_[4][posj] += choosec(orbit_counts.per_vertex_[0][j] - 1, 2);
          
          orbit_counts.per_edge_[6][posj] = orbit_counts.per_vertex_[3][i] - orbit_counts.per_edge_[1][posj];
          orbit_counts.per_edge_rev_[6][posj] = orbit_counts.per_vertex_[3][j] - orbit_counts.per_edge_[1][posj];
          
          orbit_counts.per_edge_[8][posj] = orbit_counts.per_edge_[1][posj] * (orbit_counts.per_vertex_[0][i] - 2);
          orbit_counts.per_edge_rev_[8][posj] = orbit_counts.per_edge_[1][posj] *  (orbit_counts.per_vertex_[0][j] - 2);

          orbit_counts.per_edge_[10][posj] = choosec(orbit_counts.per_edge_[1][posj], 2);

          orbit_counts.per_vertex_[9][i] += orbit_counts.per_vertex_[3][j] - orbit_counts.per_edge_[1][posj];
          orbit_counts.per_vertex_[10][i] += orbit_counts.per_edge_[1][posj] * (orbit_counts.per_vertex_[0][j] - 2);
          orbit_counts.per_vertex_[13][j] += choosec(orbit_counts.per_edge_[1][posj], 2);
          
          for (VertexIdx posk = posj+1; posk < gout->offsets[i+1]; posk++) // loop over another out-neighbor of i, that is "ahead" of j in list of out-neighbors
          {
                k = gout->nbors[posk];
                loc = gout->getEdgeBinary(j,k); 
                if(loc != -1) // (j, k) is present
                {
                    orbit_counts.per_vertex_[12][i] += orbit_counts.per_edge_[1][loc] - 1;
                    orbit_counts.per_vertex_[12][j] += orbit_counts.per_edge_[1][posk] - 1;
                    orbit_counts.per_vertex_[12][k] += orbit_counts.per_edge_[1][posj] - 1;

                    orbit_counts.per_edge_[7][posj] += orbit_counts.per_vertex_[0][k]-2;
                    orbit_counts.per_edge_[7][posk] += orbit_counts.per_vertex_[0][j]-2;
                    orbit_counts.per_edge_[7][loc] += orbit_counts.per_vertex_[0][i]-2;
                    
                    orbit_counts.per_edge_[9][posj] += orbit_counts.per_edge_[1][posk] - 1;
                    orbit_counts.per_edge_rev_[9][posj] += orbit_counts.per_edge_[1][loc] - 1;
                    orbit_counts.per_edge_[9][posk] += orbit_counts.per_edge_[1][posj] - 1;
                    orbit_counts.per_edge_rev_[9][posk] += orbit_counts.per_edge_[1][loc] - 1;
                    orbit_counts.per_edge_[9][loc] += orbit_counts.per_edge_[1][posj] - 1;
                    orbit_counts.per_edge_rev_[9][loc] += orbit_counts.per_edge_[1][posk] - 1;

                }
          }
          
          // E_2(i) : we don't use it for node orbit counts - Fix Later
          // for (VertexIdx posk = gout->offsets[i]; posk < gout->offsets[i+1]; posk++) //looping over out-neighbors of i
          // {
          //   if (posk == posj)
          //       continue;
          //   k = gout->nbors[posk];
          //   orbit_counts.per_edge_[2][posj] += orbit_counts.per_vertex_[0][k] - 1;
          // }
          // for (VertexIdx posk = gin->offsets[i]; posk < gin->offsets[i+1]; posk++) //looping over in-neighbors of i
          // {
          //   k = gin->nbors[posk];
          //   orbit_counts.per_edge_[2][posj] += orbit_counts.per_vertex_[0][k] - 1;
          // }
          // orbit_counts.per_edge_[2][posj] -= orbit_counts.per_edge_[1][posj];

          // loc = gin->getEdgeBinary(j, i); //(i,j) ?
          // for (VertexIdx posk = gout->offsets[j]; posk < gout->offsets[j+1]; posk++)
          // {
          //   k = gout->nbors[posk];
          //   orbit_counts.per_edge_rev_[2][posj] += orbit_counts.per_vertex_[0][k] - 1;
          // }
          // for (VertexIdx posk = gin->offsets[j]; posk < gin->offsets[j+1]; posk++)
          // {
          //   if (posk == loc)
          //       continue;
          //   k = gin->nbors[posk];   
          //   orbit_counts.per_edge_rev_[2][posj] += orbit_counts.per_vertex_[0][k] - 1;
          // }
          // orbit_counts.per_edge_rev_[2][posj] -= orbit_counts.per_edge_[1][posj];

        }

       for (EdgeIdx posj = gin->offsets[i]; posj < gin->offsets[i+1]; posj++)   // loop over in-neighbor of i
       {
          j = gin->nbors[posj];
          loc = gout->getEdgeBinary(j, i);
          orbit_counts.per_vertex_[9][i] += orbit_counts.per_vertex_[3][j] - orbit_counts.per_edge_[1][loc];
          orbit_counts.per_vertex_[10][i] += orbit_counts.per_edge_[1][loc] * (orbit_counts.per_vertex_[0][j] - 2);
          orbit_counts.per_vertex_[13][j] += (orbit_counts.per_edge_[1][loc] * (orbit_counts.per_edge_[1][loc] - 1)) / 2;
        }
    }
}



// Four-clique counter
// Input: The out-DAG of a graph sorted by ID (It is *critical* that DAG is sorted by ID, not be degree.)
// Output: The number of 4-cliques

// This is a function that counts Node appearances in orbits 14 (the only orbit in a 4-clique)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG) and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts

void fourCliqueOrbitCounterNodeOnly(CGraph *gout, OrbitInfo &orbit_counts)
{
    VertexIdx *triends = new VertexIdx[gout->nVertices+1]; // array to store triangle ends

    EdgeIdx edge_pos_ij, edge_pos_ik, edge_pos_il, edge_pos_jk, edge_pos_jl, edge_pos_kl;

    for (VertexIdx i=0; i < gout->nVertices; ++i) // loop over vertices
    {
        for (VertexIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; ++posj) // loop over out-neighbors of i
        {
            VertexIdx j = gout->nbors[posj]; // j is current out-neighbor
            VertexIdx count = 0;
            for (VertexIdx posk = posj+1; posk < gout->offsets[i+1]; ++posk) // loop over another out-neighbor of i, that is "ahead" of j in list of out-neighbors
            {
                VertexIdx k = gout->nbors[posk]; // k is next out-neighbor
                edge_pos_jk = gout->getEdgeBinary(j, k);
                if (edge_pos_jk != -1) // check if edge (j,k) is present
                {
                    triends[count] = k;  // so (i,j,k) form a triangle. we store the fact that k forms a triangle with edge (i,j) in digraph gout

                    ++count;
                }
            }
            for (VertexIdx posk = 0; posk < count; ++posk) // loop over all pairs of triangles formed by (i,j)
            {
                VertexIdx k = triends[posk]; // k is vertex as index posk in triends

                VertexIdx degk = gout->offsets[k+1] - gout->offsets[k]; // gettting degree of k in gout
                VertexIdx remaining = count-posk; // number of vertices that k needs to be checked with

                if (true || degk >= remaining)
                {
                    // We will search all other vertices in triends in k's adj list
                    for (VertexIdx posell = posk+1; posell < count; ++posell)
                    {
                        VertexIdx ell = triends[posell];
                        
                        edge_pos_kl = gout->getEdgeBinary(k, ell);

                        if (edge_pos_kl != -1) // (k,ell) is an end, thus (i,j,k,ell) form a 4-clique
                        {                            
                            orbit_counts.per_vertex_[14][i]++;
                            orbit_counts.per_vertex_[14][j]++;
                            orbit_counts.per_vertex_[14][k]++;
                            orbit_counts.per_vertex_[14][ell]++;

                        }
                    }
                }
                else
                {
                    // We will search all vertices in k's adj list in the remaining portion of triends
                    for (EdgeIdx posell = gout->offsets[k]; posell < gout->offsets[k+1]; posell++)
                    {
                        VertexIdx ell = gout->nbors[posell];
                        if (binarySearch(triends+posk+1,count-posk-1,ell) != -1)
                        {                        
                            orbit_counts.per_vertex_[14][i]++;
                            orbit_counts.per_vertex_[14][j]++;
                            orbit_counts.per_vertex_[14][k]++;
                            orbit_counts.per_vertex_[14][ell]++;

                        }
                    }
                }
            }
        }
    }
  delete [] triends;
}



 
// Four-clique counter
// Input: The out-DAG of a graph sorted by ID (It is *critical* that DAG is sorted by ID, not be degree.)
// Output: The number of 4-cliques

// This is a function that counts Node appearances in orbits 14 (the only orbit in a 4-clique) and also edge appearances in edge orbits 11 (the only edge orbit in 4-clique)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG) and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts

void fourCliqueOrbitCounter(CGraph *gout, OrbitInfo &orbit_counts)
{
    VertexIdx *triends = new VertexIdx[gout->nVertices+1]; // array to store triangle ends
    
    EdgeIdx *edge_pos_i_triend = new VertexIdx[gout->nVertices+1]; // array to store triangle ends
    EdgeIdx *edge_pos_j_triend = new VertexIdx[gout->nVertices+1]; // array to store triangle ends
    

    EdgeIdx edge_pos_ij, edge_pos_ik, edge_pos_il, edge_pos_jk, edge_pos_jl, edge_pos_kl;

    for (VertexIdx i=0; i < gout->nVertices; ++i) // loop over vertices
    {
        for (VertexIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; ++posj) // loop over out-neighbors of i
        {
            VertexIdx j = gout->nbors[posj]; // j is current out-neighbor
            edge_pos_ij = posj;

            VertexIdx count = 0;
            for (VertexIdx posk = posj+1; posk < gout->offsets[i+1]; ++posk) // loop over another out-neighbor of i, that is "ahead" of j in list of out-neighbors
            {
                VertexIdx k = gout->nbors[posk]; // k is next out-neighbor
                edge_pos_jk = gout->getEdgeBinary(j, k);
                if (edge_pos_jk != -1) // check if edge (j,k) is present
                {
                    triends[count] = k;  // so (i,j,k) form a triangle. we store the fact that k forms a triangle with edge (i,j) in digraph gout
                    edge_pos_i_triend[count] = posk;
                    edge_pos_j_triend[count] = edge_pos_jk;

                    ++count;
                }
            }
            for (VertexIdx posk = 0; posk < count; ++posk) // loop over all pairs of triangles formed by (i,j)
            {
                VertexIdx k = triends[posk]; // k is vertex as index posk in triends
                
                edge_pos_ik = edge_pos_i_triend[posk];
                edge_pos_jk = edge_pos_j_triend[posk];

                VertexIdx degk = gout->offsets[k+1] - gout->offsets[k]; // gettting degree of k in gout
                VertexIdx remaining = count-posk; // number of vertices that k needs to be checked with

                if (true || degk >= remaining)
                {
                    // We will search all other vertices in triends in k's adj list
                    for (VertexIdx posell = posk+1; posell < count; ++posell)
                    {
                        VertexIdx ell = triends[posell];

                        edge_pos_il = edge_pos_i_triend[posell];
                        edge_pos_jl = edge_pos_j_triend[posell];

                        
                        edge_pos_kl = gout->getEdgeBinary(k, ell);

                        if (edge_pos_kl != -1) // (k,ell) is an end, thus (i,j,k,ell) form a 4-clique
                        {
                            
                            orbit_counts.per_vertex_[14][i]++;
                            orbit_counts.per_vertex_[14][j]++;
                            orbit_counts.per_vertex_[14][k]++;
                            orbit_counts.per_vertex_[14][ell]++;

                            orbit_counts.per_edge_[11][edge_pos_ij]++;
                            orbit_counts.per_edge_[11][edge_pos_ik]++;
                            orbit_counts.per_edge_[11][edge_pos_il]++;
                            orbit_counts.per_edge_[11][edge_pos_jk]++;
                            orbit_counts.per_edge_[11][edge_pos_jl]++;
                            orbit_counts.per_edge_[11][edge_pos_kl]++;
                        }
                    }
                }
                else
                {
                    // We will search all vertices in k's adj list in the remaining portion of triends
                    for (EdgeIdx posell = gout->offsets[k]; posell < gout->offsets[k+1]; posell++)
                    {
                        VertexIdx ell = gout->nbors[posell];
                        edge_pos_kl = posell;
                        if (binarySearch(triends+posk+1,count-posk-1,ell) != -1)
                        {
                            edge_pos_il = gout->getEdgeBinary(i, ell);                            
                            edge_pos_jl = gout->getEdgeBinary(j, ell);
                            
                            orbit_counts.per_vertex_[14][i]++;
                            orbit_counts.per_vertex_[14][j]++;
                            orbit_counts.per_vertex_[14][k]++;
                            orbit_counts.per_vertex_[14][ell]++;

                            orbit_counts.per_edge_[11][edge_pos_ij]++;
                            orbit_counts.per_edge_[11][edge_pos_ik]++;
                            orbit_counts.per_edge_[11][edge_pos_il]++;
                            orbit_counts.per_edge_[11][edge_pos_jk]++;
                            orbit_counts.per_edge_[11][edge_pos_jl]++;
                            orbit_counts.per_edge_[11][edge_pos_kl]++;
                        }
                    }
                }
            }
        }
    }
  delete [] triends;
}

#endif



