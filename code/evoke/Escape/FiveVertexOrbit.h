#ifndef ESCAPE_FIVEVERTEXORBIT_H_
#define ESCAPE_FIVEVERTEXORBIT_H_

#include <iostream>
#include <algorithm>
#include <ctime>

#include "Escape/ErrorCode.h"
#include "Escape/Graph.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/Utils.h"

using namespace std;
using namespace Escape;


// This is a function that counts Node appearances in node orbit of tree patterns with 5 vertices
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts

void FiveTreeOrbitCounter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    VertexIdx j;
    for (VertexIdx i=0; i < gout->nVertices; i++) // loop over vertices
    {
        orbit_counts.per_vertex_[16][i] = orbit_counts.per_vertex_[4][i] * (orbit_counts.per_vertex_[0][i]-1)
        - orbit_counts.per_vertex_[10][i] - 2 * orbit_counts.per_vertex_[8][i];

        orbit_counts.per_vertex_[17][i] = choosec(orbit_counts.per_vertex_[1][i], 2)
        - orbit_counts.per_vertex_[3][i] - orbit_counts.per_vertex_[6][i]
        - orbit_counts.per_vertex_[8][i] - orbit_counts.per_vertex_[10][i];

        orbit_counts.per_vertex_[23][i] = choosec(orbit_counts.per_vertex_[0][i], 4);
        
        for (EdgeIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; posj++) // loop over out-neighbors of i
        {   
            j = gout -> nbors[posj]; 
            orbit_counts.per_vertex_[15][i] += orbit_counts.per_vertex_[4][j];
            orbit_counts.per_vertex_[18][i] += orbit_counts.per_vertex_[6][j];
            orbit_counts.per_vertex_[19][i] += orbit_counts.per_vertex_[5][j];
            orbit_counts.per_vertex_[20][i] += (orbit_counts.per_vertex_[0][i] - 1) * choosec(orbit_counts.per_vertex_[0][j] - 1, 2);
            orbit_counts.per_vertex_[21][i] += (orbit_counts.per_vertex_[0][j] - 1) * choosec(orbit_counts.per_vertex_[0][i] - 1, 2);
            orbit_counts.per_vertex_[22][i] += choosec(orbit_counts.per_vertex_[0][j] - 1, 3);
        }
        for (EdgeIdx posj = gin->offsets[i]; posj < gin->offsets[i+1]; posj++)   // loop over in-neighbor of i
        {
            j = gin->nbors[posj];
            orbit_counts.per_vertex_[15][i] += orbit_counts.per_vertex_[4][j];
            orbit_counts.per_vertex_[18][i] += orbit_counts.per_vertex_[6][j];
            orbit_counts.per_vertex_[19][i] += orbit_counts.per_vertex_[5][j];
            orbit_counts.per_vertex_[20][i] += (orbit_counts.per_vertex_[0][i] - 1) * choosec(orbit_counts.per_vertex_[0][j] - 1, 2);
            orbit_counts.per_vertex_[21][i] += (orbit_counts.per_vertex_[0][j] - 1) * choosec(orbit_counts.per_vertex_[0][i] - 1, 2);
            orbit_counts.per_vertex_[22][i] += choosec(orbit_counts.per_vertex_[0][j] - 1, 3);
        }
        orbit_counts.per_vertex_[15][i] -= orbit_counts.per_vertex_[5][i]
        + 2 * orbit_counts.per_vertex_[8][i]
        + 2 * orbit_counts.per_vertex_[11][i];

        orbit_counts.per_vertex_[18][i] -= 3 * orbit_counts.per_vertex_[7][i]
                                        + orbit_counts.per_vertex_[10][i];
        
        orbit_counts.per_vertex_[19][i] -= orbit_counts.per_vertex_[4][i]
        + orbit_counts.per_vertex_[5][i]
        + orbit_counts.per_vertex_[10][i];
        
        orbit_counts.per_vertex_[20][i] -= orbit_counts.per_vertex_[10][i];
        orbit_counts.per_vertex_[21][i] -= 2*orbit_counts.per_vertex_[11][i];
    }

}

// This is a function that counts Node appearances in the only node orbit 5-cycle (R_34)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG), and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for the only node orbit 5-cycle (R_34)

void FiveCycleOrbitCounter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{

  VertexIdx i,j,k,ell;

  VertexIdx *wedge_count = new VertexIdx[gin->nVertices+1];
   
  VertexIdx *wedge_count_type1 = new VertexIdx[gin->nVertices+1]; // in-out wedge from ell to i
  VertexIdx *wedge_count_type2 = new VertexIdx[gin->nVertices+1]; // in-in wedge between ell and i
  VertexIdx *wedge_count_type3 = new VertexIdx[gin->nVertices+1]; // in-out wedge from i to ell

  VertexIdx *three_path_counts = new VertexIdx[gin->nVertices+1]; // three paths ending at i (counts for each ell whicl is the other end)
   
  VertexIdx *triangle_inin_edge = new VertexIdx[gin->nEdges]; // traingles on each edge with in edges from the third node


  for (i=0; i < gin->nVertices; i++)
  {
       wedge_count[i] = 0;
       wedge_count_type1[i] = 0;
       wedge_count_type2[i] = 0;
       wedge_count_type3[i] = 0;
       three_path_counts[i] = 0;
  }
  for (i=0; i < gin->nEdges; i++)
  {
       triangle_inin_edge[i] = 0;
  }


  // looping over all triangles
  for (i=0; i < gin->nVertices; ++i) // loop over vertices
  {
    for (EdgeIdx pos = gout->offsets[i]; pos < gout->offsets[i+1]; ++pos) // loop over out-neighbors of i
    {
      j = gout->nbors[pos];
      for (EdgeIdx next = pos+1; next < gout->offsets[i+1]; ++next) // loop over another out-neighbor of i that is "ahead" of j
      {
        k = gout->nbors[next];
        EdgeIdx pos_jk = gout->getEdgeBinary(j,k);
        if(pos_jk == -1)
          continue;
        triangle_inin_edge[pos_jk]++; // i has edges to j and k, so for edge (j,k) i is a third node for an inin triangle
      }
    }
  }



  // computing wedge counts,
  // looping over three paths to count five cycles for nodes except the on on the wedge,
  // looping over wedges to count five cucles for the node on the wedge,
  // and finally clearing wedge counts and three path counts
  for (i=0; i < gin->nVertices; ++i) // loop over vertices
  {
    for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
    {
      j = gin->nbors[pos]; // j is current in-neighbor
      for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note that this gives an in-out wedge
      {
        k = gin->nbors[next];  // i <- j <- k is a type 3wedge
        wedge_count_type3[k]++;
        wedge_count[k]++;
      }

      for (EdgeIdx next = gout->offsets[j]; next < gout->offsets[j+1]; ++next) // loop over out-neighbors of j, note that this gives an out-out wedge
      {
        k = gout->nbors[next]; // i <- j -> k is a type 2 wedge
        if (k == i)
          continue;
        wedge_count_type2[k]++;
        wedge_count[k]++;
      }
    }

    // loop over inout wedges starting at i
    for (EdgeIdx pos = gout->offsets[i]; pos < gout->offsets[i+1]; ++pos) // loop over out neighbors of i
    {
      j = gout->nbors[pos];
      for (EdgeIdx next = gout->offsets[j]; next < gout->offsets[j+1]; ++next) // loop over out neighbors of j, note that this gives an in-out wedge
      {
        k = gout->nbors[next]; // i -> j -> k is type 1 wedge
        wedge_count_type1[k]++;
        wedge_count[k]++;
      }
    }


    // generate three paths ending at i
    for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
    {
      j = gin->nbors[pos];
      for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j
      {
        k = gin->nbors[next]; // i <- j <- k is wedge
        for (EdgeIdx next2 = gout->offsets[k]; next2 < gout->offsets[k+1]; ++next2) // loop over out-neighbors of k
        {
          ell = gout->nbors[next2]; // i <- j <- k -> ell is three path
          if (ell == j || ell == i)
            continue;
          EdgeIdx addition = wedge_count[ell];
                   
          if (gout->getEdgeBinary(i,k) != -1 || gout->getEdgeBinary(k,i) != -1)
            addition--;
          if (gout->getEdgeBinary(j,ell) != -1 || gout->getEdgeBinary(ell,j) != -1)
          addition--;

          three_path_counts[ell]++;

          orbit_counts.per_vertex_[34][i] += addition;
          orbit_counts.per_vertex_[34][j] += addition;
          orbit_counts.per_vertex_[34][k] += addition;
          orbit_counts.per_vertex_[34][ell] += addition;
        }
      }
    }

    //continue here at home
    //loop over wedges to count five cycles for the middle node on the wedge using three paths counts 
    for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
    {
      j = gin->nbors[pos]; // j is current in-neighbor
      for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note that this gives an in-out wedge
      {
        k = gin->nbors[next];  // i <- j <- k is a type 3 wedge
        EdgeIdx pos_kj = gout->getEdgeBinary(k,j);
        orbit_counts.per_vertex_[34][j] += three_path_counts[k];
        orbit_counts.per_vertex_[34][j] -= triangle_inin_edge[pos_kj];
      }

      for (EdgeIdx next = gout->offsets[j]; next < gout->offsets[j+1]; ++next) // loop over out-neighbors of j, note that this gives an out-out wedge
      {
        k = gout->nbors[next]; // i <- j -> k is a type 2 wedge
        if (k == i)
          continue;
        orbit_counts.per_vertex_[34][j] += three_path_counts[k];
        orbit_counts.per_vertex_[34][j] -= wedge_count_type3[j] - (gout->getEdgeBinary(k, i) != -1);
        orbit_counts.per_vertex_[34][j] -= triangle_inin_edge[next];
      }
    }

    // loop over inout wedges starting at i
    for (EdgeIdx pos = gout->offsets[i]; pos < gout->offsets[i+1]; ++pos) // loop over out neighbors of i
    {
      j = gout->nbors[pos];
      for (EdgeIdx next = gout->offsets[j]; next < gout->offsets[j+1]; ++next) // loop over out neighbors of j, note that this gives an in-out wedge
      {
        k = gout->nbors[next]; // i -> j -> k is type 1 wedge
        orbit_counts.per_vertex_[34][j] += three_path_counts[k];
      }
    }


    // clear wedge_count
    // loop over inout wedges ending at i
    for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
    {
      j = gin->nbors[pos]; // j is current in-neighbor
      for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives an inout wedge
      {
        k = gin->nbors[next];  // i <- j <- k is type 3 wedge
        wedge_count_type3[k] = 0;
        wedge_count[k] = 0;
      }

      for (EdgeIdx next = gout->offsets[j]; next < gout->offsets[j+1]; ++next) // loop over out-neighbors of j, note that this gives an out-out wedge
      {
        k = gout->nbors[next]; // i <- j -> k is a type 2wedge
        wedge_count_type2[k] = 0;
        wedge_count[k] = 0;
      }
    }

    // loop over inout wedges starting at i
    for (EdgeIdx pos = gout->offsets[i]; pos < gout->offsets[i+1]; ++pos) // loop over out-neighbors of i
    {
      j = gout->nbors[pos];
      for (EdgeIdx next = gout->offsets[j]; next < gout->offsets[j+1]; ++next) // loop over out-neighbors of i
      {
        k = gout->nbors[next]; // i -> j -> k isa type 1 wedge
        wedge_count_type2[k] = 0;
        wedge_count[k] = 0;
      }
    }

    // clearing three path counts
    for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
    {
      j = gin->nbors[pos];
      for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j
      {
        k = gin->nbors[next]; // i <- j <- k is wedge
        for (EdgeIdx next2 = gout->offsets[k]; next2 < gout->offsets[k+1]; ++next2) // loop over out-neighbors of k
        {
          ell = gout->nbors[next2]; // i <- j <- k -> ell is three path
          if (ell == j || ell == i)
            continue;
          three_path_counts[ell] = 0;
        }
      }
    }

  }

  delete [] wedge_count;
  delete [] wedge_count_type1;
  delete [] wedge_count_type2;
  delete [] wedge_count_type3;
  delete [] three_path_counts;
  delete [] triangle_inin_edge;

}

// This is a function that counts Node appearances in node orbits of the three wedge collision pattern (R_49 and R_50)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbits of the three wedge collision pattern (R_49 and R_50)

void WedgeCollisionOrbitCounter(CGraph *cg, CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    VertexIdx i,j,k;

    VertexIdx *wedge_count = new VertexIdx[cg->nVertices+1];
    VertexIdx *wedge_count_cp = new VertexIdx[cg->nVertices+1]; // a copy of wedge count that we don't erase

    for (i=0; i < cg->nVertices; i++)
    {
       wedge_count[i] = 0;
       wedge_count_cp[i] = 0;
    }

     for (i=0; i < cg->nVertices; ++i) // loop over vertices
     { 
        VertexIdx degi = cg->offsets[i+1] - cg->offsets[i];
 
        // loop over wedges to populate the ends of the wedges
        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; ++pos) // loop over neighbors of i
        {
            j = cg->nbors[pos]; // j is current neighbor
            for (EdgeIdx next = cg->offsets[j]; next < cg->offsets[j+1]; ++next) // loop over neighbors of j, note this gives a wedge centered at j
            {
                k = cg->nbors[next];  // i -- j -- k is wedge centered at j
                if (k <= i)
                    continue;
                wedge_count[k]++;
                wedge_count_cp[k] = wedge_count[k];
            }
        }

        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; ++pos) // loop over in-neighbors of i
        {
            j = cg->nbors[pos]; // j is current neighbor
            for (EdgeIdx next = cg->offsets[j]; next < cg->offsets[j+1]; ++next) // loop over neighbors of j, note this gives a wedge centered at j
            {
                k = cg->nbors[next];  // i -- j -- k is a wedge centered at j
                if(k <= i)
                  continue;
                orbit_counts.per_vertex_[49][j] += choosec(wedge_count_cp[k]-1, 2);
                orbit_counts.per_vertex_[50][i] += choosec(wedge_count[k], 3);
                orbit_counts.per_vertex_[50][k] += choosec(wedge_count[k], 3);
                wedge_count[k] = 0;
            }
        }
      }


  delete [] wedge_count;
  delete [] wedge_count_cp;
}


// This is a function that counts Node appearances in node orbits R_51
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbits R_51

void Pattern51OrbitCounter(CGraph *cg, CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
   VertexIdx i,j,k;
   EdgeIdx pos_ji, pos_jk, pos_kj, pos_ik;

   VertexIdx *triangle_on_ij = new VertexIdx[gout->nVertices+1];
   VertexIdx *triangle_on_jk = new VertexIdx[gout->nVertices+1];

   for (i=0; i < gout->nVertices; i++) //initialize all triangle on wedge count values to 0
   {
       triangle_on_ij[i] = 0;
       triangle_on_jk[i] = 0;
   }

   for (i=0; i < gin->nVertices; ++i) // loop over vertices
   {
        //get triangle on wedge counts
       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           pos_ji = gout->getEdgeBinary(j,i);


           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k>=i)   // break ties to prevent overcount
                   continue;
               triangle_on_ij[k] += orbit_counts.per_edge_[1][pos_ji];
               triangle_on_jk[k] += orbit_counts.per_edge_[1][next]; // next is currently pos_jk
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               pos_kj = gout->getEdgeBinary(k,j);

               triangle_on_ij[k] += orbit_counts.per_edge_[1][pos_ji];
               triangle_on_jk[k] += orbit_counts.per_edge_[1][pos_kj];
           }
       }

       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           pos_ji = gout->getEdgeBinary(j,i);

           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k>=i)   // break ties to prevent overcount
                   continue;

               orbit_counts.per_vertex_[51][i] += triangle_on_jk[k] - orbit_counts.per_edge_[1][next];
               orbit_counts.per_vertex_[51][j] += triangle_on_jk[k] - orbit_counts.per_edge_[1][next];

               orbit_counts.per_vertex_[51][j] += triangle_on_ij[k] - orbit_counts.per_edge_[1][pos_ji];
               orbit_counts.per_vertex_[51][k] += triangle_on_ij[k] - orbit_counts.per_edge_[1][pos_ji];

           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
              k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
              pos_kj = gout->getEdgeBinary(k,j);

              orbit_counts.per_vertex_[51][i] += triangle_on_jk[k] - orbit_counts.per_edge_[1][pos_kj];
              orbit_counts.per_vertex_[51][j] += triangle_on_jk[k] - orbit_counts.per_edge_[1][pos_kj];

              orbit_counts.per_vertex_[51][j] += triangle_on_ij[k] - orbit_counts.per_edge_[1][pos_ji];
              orbit_counts.per_vertex_[51][k] += triangle_on_ij[k] - orbit_counts.per_edge_[1][pos_ji];

           }
       }

       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
              k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
              if (k>=i)   // break ties to prevent overcount
                   continue;
              
              triangle_on_jk[k] = 0;
              triangle_on_ij[k] = 0;
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
              k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j

              triangle_on_jk[k] = 0;
              triangle_on_ij[k] = 0;
           }
      }
  }

  for (i=0; i < gout->nVertices; i++)
  {
    orbit_counts.per_vertex_[51][i] -= 2 * orbit_counts.per_vertex_[12][i] + 2 * orbit_counts.per_vertex_[13][i];
  }


  delete [] triangle_on_ij;
  delete [] triangle_on_jk;
}


// This is a function that counts Node appearances in node orbit for remaining orbits of patterns with 5 vertices
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts

void OtherFiveOrbitsCounter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    VertexIdx j, k;
    EdgeIdx loc;
    for (VertexIdx i=0; i < gout->nVertices; i++) // loop over vertices
    {
        orbit_counts.per_vertex_[33][i] = orbit_counts.per_vertex_[3][i] * choosec(orbit_counts.per_vertex_[0][i]-2, 2);
        orbit_counts.per_vertex_[38][i] = orbit_counts.per_vertex_[8][i] * (orbit_counts.per_vertex_[0][i] - 2)
                                        - orbit_counts.per_vertex_[13][i];

        orbit_counts.per_vertex_[44][i] = choosec(orbit_counts.per_vertex_[3][i], 2) - orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[47][i] = orbit_counts.per_vertex_[12][i] * (orbit_counts.per_vertex_[0][i] - 2)
                                        - 3 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[58][i] = orbit_counts.per_vertex_[14][i] * (orbit_counts.per_vertex_[0][i] - 3);

        for (EdgeIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; posj++) // loop over out-neighbors of i
        { 
            j = gout->nbors[posj];
            orbit_counts.per_vertex_[24][i] += orbit_counts.per_vertex_[10][j];
            orbit_counts.per_vertex_[27][i] += orbit_counts.per_vertex_[9][j];
            orbit_counts.per_vertex_[28][i] += (orbit_counts.per_vertex_[0][i] - 1) * (orbit_counts.per_vertex_[3][j]); 
            orbit_counts.per_vertex_[30][i] += (orbit_counts.per_vertex_[0][j] - 1) * (orbit_counts.per_vertex_[3][i]);
            orbit_counts.per_vertex_[31][i] += (orbit_counts.per_vertex_[0][j] - 1) * orbit_counts.per_vertex_[3][j];
            orbit_counts.per_vertex_[35][i] += orbit_counts.per_vertex_[8][j];
            orbit_counts.per_vertex_[37][i] += orbit_counts.per_edge_[5][posj] * (orbit_counts.per_vertex_[0][j] - 2);
            orbit_counts.per_vertex_[37][j] += orbit_counts.per_edge_[5][posj] * (orbit_counts.per_vertex_[0][i] - 2);
            orbit_counts.per_vertex_[39][i] += orbit_counts.per_vertex_[13][j];
            orbit_counts.per_vertex_[40][i] += orbit_counts.per_edge_rev_[9][posj] * (orbit_counts.per_vertex_[0][j] - 3);
            orbit_counts.per_vertex_[40][j] += orbit_counts.per_edge_[9][posj] * (orbit_counts.per_vertex_[0][i] - 3);
            orbit_counts.per_vertex_[41][i] += choosec(orbit_counts.per_edge_[1][posj], 2) * (orbit_counts.per_vertex_[0][j] - 3);
            orbit_counts.per_vertex_[41][j] += choosec(orbit_counts.per_edge_[1][posj], 2) * (orbit_counts.per_vertex_[0][i] - 3);
            orbit_counts.per_vertex_[42][i] += choosec(orbit_counts.per_edge_[1][posj], 2) * (orbit_counts.per_vertex_[0][i] - 3);
            orbit_counts.per_vertex_[42][j] += choosec(orbit_counts.per_edge_[1][posj], 2) * (orbit_counts.per_vertex_[0][j] - 3);
            orbit_counts.per_vertex_[45][i] += orbit_counts.per_vertex_[12][j];

            orbit_counts.per_vertex_[55][i] += choosec(orbit_counts.per_edge_[1][posj], 3);
            orbit_counts.per_vertex_[55][j] += choosec(orbit_counts.per_edge_[1][posj], 3);

            orbit_counts.per_vertex_[56][i] += orbit_counts.per_vertex_[14][j];
            orbit_counts.per_vertex_[57][i] += orbit_counts.per_edge_[11][posj] * (orbit_counts.per_vertex_[0][j] - 3);
            orbit_counts.per_vertex_[57][j] += orbit_counts.per_edge_[11][posj] * (orbit_counts.per_vertex_[0][i] - 3);

            orbit_counts.per_vertex_[60][i] += orbit_counts.per_edge_rev_[9][posj] * (orbit_counts.per_edge_[1][posj] - 1);
            orbit_counts.per_vertex_[60][j] += orbit_counts.per_edge_[9][posj] * (orbit_counts.per_edge_[1][posj]  - 1);

            orbit_counts.per_vertex_[67][i] += orbit_counts.per_edge_[11][posj] * (orbit_counts.per_edge_[1][posj] - 2);
            orbit_counts.per_vertex_[67][j] += orbit_counts.per_edge_[11][posj] * (orbit_counts.per_edge_[1][posj] - 2);
            
            for (VertexIdx posk = posj+1; posk < gout->offsets[i+1]; posk++) // loop over another out-neighbor of i, that is "ahead" of j in list of out-neighbors
            {
                k = gout->nbors[posk];
                loc = gout->getEdgeBinary(j,k);
                if (loc != -1)
                {
                
                    orbit_counts.per_vertex_[25][i] += (orbit_counts.per_vertex_[0][j] - 2) * (orbit_counts.per_vertex_[0][k] - 2);
                    orbit_counts.per_vertex_[25][j] += (orbit_counts.per_vertex_[0][i] - 2) * (orbit_counts.per_vertex_[0][k] - 2);
                    orbit_counts.per_vertex_[25][k] += (orbit_counts.per_vertex_[0][i] - 2) * (orbit_counts.per_vertex_[0][j] - 2);
                
                    orbit_counts.per_vertex_[26][i] += (orbit_counts.per_vertex_[0][i] - 2)
                                                    * ((orbit_counts.per_vertex_[0][j] - 2) + (orbit_counts.per_vertex_[0][k] - 2));
                    orbit_counts.per_vertex_[26][j] += (orbit_counts.per_vertex_[0][j] - 2)
                                                    * ((orbit_counts.per_vertex_[0][i] - 2) + (orbit_counts.per_vertex_[0][k] - 2));
                    orbit_counts.per_vertex_[26][k] += (orbit_counts.per_vertex_[0][k] - 2)
                                                    * ((orbit_counts.per_vertex_[0][i] - 2) + (orbit_counts.per_vertex_[0][j] - 2));

                    orbit_counts.per_vertex_[29][i] += orbit_counts.per_vertex_[1][j] + orbit_counts.per_vertex_[1][k];
                    orbit_counts.per_vertex_[29][j] += orbit_counts.per_vertex_[1][i] + orbit_counts.per_vertex_[1][k];
                    orbit_counts.per_vertex_[29][k] += orbit_counts.per_vertex_[1][i] + orbit_counts.per_vertex_[1][j];

                    orbit_counts.per_vertex_[32][i] += choosec(orbit_counts.per_vertex_[0][j]-2, 2) + choosec(orbit_counts.per_vertex_[0][k]-2, 2);
                    orbit_counts.per_vertex_[32][j] += choosec(orbit_counts.per_vertex_[0][i]-2, 2) + choosec(orbit_counts.per_vertex_[0][k]-2, 2);
                    orbit_counts.per_vertex_[32][k] += choosec(orbit_counts.per_vertex_[0][i]-2, 2) + choosec(orbit_counts.per_vertex_[0][j]-2, 2);

                    orbit_counts.per_vertex_[43][i] += orbit_counts.per_vertex_[3][j] - 1 + orbit_counts.per_vertex_[3][k] - 1;
                    orbit_counts.per_vertex_[43][j] += orbit_counts.per_vertex_[3][i] - 1 + orbit_counts.per_vertex_[3][k] - 1;
                    orbit_counts.per_vertex_[43][k] += orbit_counts.per_vertex_[3][i] - 1 + orbit_counts.per_vertex_[3][j] - 1;

                    orbit_counts.per_vertex_[46][i] += orbit_counts.per_edge_[7][loc];
                    orbit_counts.per_vertex_[46][j] += orbit_counts.per_edge_[7][posk];
                    orbit_counts.per_vertex_[46][k] += orbit_counts.per_edge_[7][posj];

                    orbit_counts.per_vertex_[48][i] += (orbit_counts.per_edge_[1][posj] - 1) * (orbit_counts.per_vertex_[0][k] - 2)
                                                    +  (orbit_counts.per_edge_[1][posk] - 1) * (orbit_counts.per_vertex_[0][j] - 2);
                    orbit_counts.per_vertex_[48][j] += (orbit_counts.per_edge_[1][posj] - 1) * (orbit_counts.per_vertex_[0][k] - 2)
                                                    +  (orbit_counts.per_edge_[1][loc] - 1) * (orbit_counts.per_vertex_[0][i] - 2);
                    orbit_counts.per_vertex_[48][k] += (orbit_counts.per_edge_[1][posk] - 1) * (orbit_counts.per_vertex_[0][j] - 2)
                                                    +  (orbit_counts.per_edge_[1][loc] - 1) * (orbit_counts.per_vertex_[0][i] - 2);

                    orbit_counts.per_vertex_[52][i] += orbit_counts.per_edge_[5][loc];
                    orbit_counts.per_vertex_[52][j] += orbit_counts.per_edge_[5][posk];
                    orbit_counts.per_vertex_[52][k] += orbit_counts.per_edge_[5][posj];

                    orbit_counts.per_vertex_[53][i] += orbit_counts.per_edge_[5][posj] + orbit_counts.per_edge_[5][posk];
                    orbit_counts.per_vertex_[53][j] += orbit_counts.per_edge_[5][posj] + orbit_counts.per_edge_[5][loc];
                    orbit_counts.per_vertex_[53][k] += orbit_counts.per_edge_[5][posk] + orbit_counts.per_edge_[5][loc];

                    orbit_counts.per_vertex_[54][i] += choosec(orbit_counts.per_edge_[1][loc] - 1, 2);
                    orbit_counts.per_vertex_[54][j] += choosec(orbit_counts.per_edge_[1][posk] - 1, 2);
                    orbit_counts.per_vertex_[54][k] += choosec(orbit_counts.per_edge_[1][posj] - 1, 2);

                    orbit_counts.per_vertex_[59][i] += orbit_counts.per_edge_[9][loc] + orbit_counts.per_edge_rev_[9][loc];
                    orbit_counts.per_vertex_[59][j] += orbit_counts.per_edge_[9][posk] + orbit_counts.per_edge_rev_[9][posk];
                    orbit_counts.per_vertex_[59][k] += orbit_counts.per_edge_[9][posj] + orbit_counts.per_edge_rev_[9][posj];

                    orbit_counts.per_vertex_[61][i] += (orbit_counts.per_edge_[1][posj] - 1) * (orbit_counts.per_edge_[1][posk] - 1);
                    orbit_counts.per_vertex_[61][j] += (orbit_counts.per_edge_[1][posj] - 1) * (orbit_counts.per_edge_[1][loc] - 1);
                    orbit_counts.per_vertex_[61][k] += (orbit_counts.per_edge_[1][posk] - 1) * (orbit_counts.per_edge_[1][loc] - 1);

                    orbit_counts.per_vertex_[65][i] += orbit_counts.per_edge_[11][loc];
                    orbit_counts.per_vertex_[65][j] += orbit_counts.per_edge_[11][posk];
                    orbit_counts.per_vertex_[65][k] += orbit_counts.per_edge_[11][posj];
                }

            }
        }
        for (EdgeIdx posj = gin->offsets[i]; posj < gin->offsets[i+1]; posj++)   // loop over in-neighbor of i
        {
            j = gin->nbors[posj];

            orbit_counts.per_vertex_[24][i] += orbit_counts.per_vertex_[10][j];
            orbit_counts.per_vertex_[27][i] += orbit_counts.per_vertex_[9][j];
            orbit_counts.per_vertex_[28][i] += (orbit_counts.per_vertex_[0][i] - 1) * (orbit_counts.per_vertex_[3][j]);
            orbit_counts.per_vertex_[30][i] += (orbit_counts.per_vertex_[0][j] - 1) * (orbit_counts.per_vertex_[3][i]);
            orbit_counts.per_vertex_[31][i] += (orbit_counts.per_vertex_[0][j] - 1) * orbit_counts.per_vertex_[3][j];
            orbit_counts.per_vertex_[35][i] += orbit_counts.per_vertex_[8][j];
            orbit_counts.per_vertex_[39][i] += orbit_counts.per_vertex_[13][j];
            orbit_counts.per_vertex_[45][i] += orbit_counts.per_vertex_[12][j];
            orbit_counts.per_vertex_[56][i] += orbit_counts.per_vertex_[14][j];
        }
        orbit_counts.per_vertex_[24][i] -= orbit_counts.per_vertex_[10][i] + 2 * orbit_counts.per_vertex_[11][i]
                                        + 2 * orbit_counts.per_vertex_[12][i];
        orbit_counts.per_vertex_[25][i] -= orbit_counts.per_vertex_[12][i];
        orbit_counts.per_vertex_[27][i] -= orbit_counts.per_vertex_[11][i] + 2 * orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[26][i] -= 2 * orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[28][i] -= 2 * orbit_counts.per_vertex_[3][i] + 2 * orbit_counts.per_vertex_[11][i]
                                        + 2 * orbit_counts.per_vertex_[12][i];
        orbit_counts.per_vertex_[29][i] -= 4 * orbit_counts.per_vertex_[3][i] + orbit_counts.per_vertex_[10][i]
                                        + 2 * orbit_counts.per_vertex_[11][i] + 2 * orbit_counts.per_vertex_[12][i]
                                        + 2 * orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[30][i] -= 2 * orbit_counts.per_vertex_[3][i] + orbit_counts.per_vertex_[10][i]
                                        + 2 * orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[31][i] -= 2 * orbit_counts.per_vertex_[3][i] + 2 * orbit_counts.per_vertex_[9][i]
                                        + orbit_counts.per_vertex_[10][i];
        orbit_counts.per_vertex_[35][i] -= 2 * orbit_counts.per_vertex_[8][i] + orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[37][i] -= 2 * orbit_counts.per_vertex_[12][i];
        orbit_counts.per_vertex_[39][i] -= 2 * orbit_counts.per_vertex_[12][i] + orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[43][i] -= 2 * orbit_counts.per_vertex_[12][i] + 2 * orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[46][i] -= orbit_counts.per_vertex_[11][i] + 3 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[48][i] -= 6 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[45][i] -= 2 * orbit_counts.per_vertex_[13][i] + 3 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[52][i] -= 2 * orbit_counts.per_vertex_[13][i];
        orbit_counts.per_vertex_[53][i] -= 2 * orbit_counts.per_vertex_[13][i] + 2 * orbit_counts.per_vertex_[12][i];
        orbit_counts.per_vertex_[56][i] -= 3 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[59][i] -= 2 * orbit_counts.per_vertex_[13][i] + 6 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[60][i] -= 6 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[61][i] -= 3 * orbit_counts.per_vertex_[14][i];
        orbit_counts.per_vertex_[65][i] -= 3 * orbit_counts.per_vertex_[14][i];
    }
}

// This is a function that counts Node appearances in node orbits R_36
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG),  and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbits R_36

void Pattern36OrbitCounter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
   VertexIdx i,j,k;
   EdgeIdx pos_ji, pos_jk, pos_kj;

   VertexIdx *neighbor_edge_out_count = new VertexIdx[gout->nVertices+1]; //stores number of edge outs of a neighbor in C4
   VertexIdx *wedge_count = new VertexIdx[gout->nVertices+1]; //stores number of wedges ending at a vertex

   for (i=0; i < gout->nVertices; i++) //initialize all neighbor_edge_out_count and wedge_count values to 0
   {
       neighbor_edge_out_count[i] = 0;
       wedge_count[i] = 0;
   }

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
               neighbor_edge_out_count[k] += max(0., orbit_counts.per_vertex_[0][j] - 2); // increaase the number of neighbor edge out counts for k by the the degree of j
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               wedge_count[k]++; // increment number of wedges ending at k
               neighbor_edge_out_count[k] += max(0.0, orbit_counts.per_vertex_[0][j] - 2); // increaase the number of neighbor edge out counts for k by the the degree of j
           }
       }



       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           pos_ji = gout->getEdgeBinary(j,i);

           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k>=i)   // break ties to prevent overcount
                   continue;

               orbit_counts.per_vertex_[36][j] += max(0.0, neighbor_edge_out_count[k] - (orbit_counts.per_vertex_[0][j] - 2));
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               pos_kj = gout->getEdgeBinary(k,j);

               orbit_counts.per_vertex_[36][j] += max(0.0, neighbor_edge_out_count[k] - (orbit_counts.per_vertex_[0][j] - 2));
           }
       }

       for (EdgeIdx pos = gin->offsets[i]; pos < gin->offsets[i+1]; ++pos) // loop over in-neighbors of i
       {
           j = gin->nbors[pos]; // j is current in-neighbor
           for (EdgeIdx next = gout->offsets[j]; next < gout -> offsets[j+1]; ++next) // loop over out-neighbors of j, note this gives an outout wedge
           {
               k = gout->nbors[next];  // i <- j -> k is outout wedge centered at j
               if (k>=i)   // break ties to prevent overcount
                   continue;

               if(wedge_count[k] > 1)
               {
                orbit_counts.per_vertex_[36][i] += choosec(wedge_count[k], 2) * max(0.0, orbit_counts.per_vertex_[0][k] - 2);
                orbit_counts.per_vertex_[36][k] += choosec(wedge_count[k], 2) * max(0.0, orbit_counts.per_vertex_[0][i] - 2);
               }
               neighbor_edge_out_count[k] = 0; //reset value of neighbor_edge_out_count
               wedge_count[k] = 0; //reset value of wedge_count
           }
           for (EdgeIdx next = gin->offsets[j]; next < gin->offsets[j+1]; ++next) // loop over in-neighbors of j, note this gives inout wedge
           {
               k = gin->nbors[next]; // i <- j <- k is inout wedge centered at j
               if(wedge_count[k] > 1)
               {
                orbit_counts.per_vertex_[36][i] += choosec(wedge_count[k], 2) * max(0.0, orbit_counts.per_vertex_[0][k] - 2);
                orbit_counts.per_vertex_[36][k] += choosec(wedge_count[k], 2) * max(0.0, orbit_counts.per_vertex_[0][i] - 2);
               }
               neighbor_edge_out_count[k] = 0; //reset value of neighbor_edge_out_count
               wedge_count[k] = 0; //reset value of wedge_count
           }
       }

   }

   for (i=0; i < gout->nVertices; i++)
   {
    orbit_counts.per_vertex_[36][i] -= orbit_counts.per_vertex_[13][i];
   }

  delete [] neighbor_edge_out_count;
  delete [] wedge_count;
}


// This is a function that counts Node appearances in node orbits of chorda wedge patterns (orbits 62, 63, 64, 68, 69)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG), a list of triangles for each edge, and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbits of chorda wedge patterns (orbits 62, 63, 64, 68, 69)

void ChordalWedgeColOrbitCounter(CGraph *cg, CGraph *gout, CGraph *gin, TriangleList *allTris, OrbitInfo &orbit_counts)
{
   VertexIdx i,j,k,ell;

   VertexIdx *wedge_count = new VertexIdx[cg->nVertices+1];

   VertexIdx *diamond_count = new VertexIdx[cg->nVertices+1];
   VertexIdx *diamond_count_copy = new VertexIdx[cg->nVertices+1];
   VertexIdx *diamond_count_i = new VertexIdx[cg->nVertices+1];

   VertexIdx *mid_diamond_count_i = new VertexIdx[cg->nVertices+1];
      
   EdgeIdx low, mid, high;

   for (i=0; i < cg->nVertices; i++)
   {
       wedge_count[i] = 0;
       diamond_count[i] = 0;
       diamond_count_copy[i] = 0;
       diamond_count_i[i] = 0;
       mid_diamond_count_i[i] = 0;
   }


   for (i=0; i < cg->nVertices; ++i) // loop over vertices
   {
        VertexIdx degi = cg->offsets[i+1] - cg->offsets[i];

        // loop over wedges to populate the ends of the wedges
        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; ++pos) // loop over neighbors of i
        {
            j = cg->nbors[pos]; // j is current neighbor
            for (EdgeIdx next = cg->offsets[j]; next < cg->offsets[j+1]; ++next) // loop over neighbors of j
            {

                k = cg->nbors[next];  // i -- j -- k is wedge centered at j
                if (k <= i)
                    continue;
                wedge_count[k]++;

            }
        } 
         

        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; ++pos) // loop over in-neighbors of i
        {
          j = cg->nbors[pos]; // j is current neighbor


            VertexIdx degj = cg->offsets[j+1] - cg->offsets[j];
            EdgeIdx edge_pos = 0;
            if ((degj < degi) || (degj==degi && j <= i))
                edge_pos = cg->getEdgeBinary(j,i);
            else
                edge_pos = pos;

            // looping over all diamonds for rdge (i, j)
            for (EdgeIdx tri_pos = allTris->trioffsets[edge_pos]; tri_pos < allTris->trioffsets[edge_pos+1]; tri_pos++) // loop over triangles on edge (i, j)
            {
                k = allTris -> triangles[tri_pos];

                VertexIdx degk = cg->offsets[k+1] - cg->offsets[k];
                VertexIdx new_pos = 0;
                if ((degk < degj) || (degk==degj && k <= j))
                    new_pos = cg->getEdgeBinary(k,j);
                else
                    new_pos = cg->getEdgeBinary(j,k);
            

                low = allTris->trioffsets[new_pos];
                high = allTris->trioffsets[new_pos+1]-1;

                while(low <= high)
                {
                     mid = (low+high)/2;
                     if (allTris->triangles[mid] == i)
                         break;
                     if (allTris->triangles[mid] > i)
                         high = mid-1;
                     else
                         low = mid+1;
                }

                for (EdgeIdx next_tri_pos = mid; next_tri_pos < allTris->trioffsets[new_pos+1]; next_tri_pos++) // loop over triangles on edge (j,k) with the third end ahead of i
                {
                    ell = allTris->triangles[next_tri_pos];

                    if (i >= ell)
                      continue;

                    diamond_count[ell]++;
                    diamond_count_copy[ell] = diamond_count[ell];
                    
                    if(j < k)
                    {
                      diamond_count_i[ell]++;
                      mid_diamond_count_i[j]++;
                      mid_diamond_count_i[k]++;

                      if(wedge_count[ell] > 2)
                      {
                        orbit_counts.per_vertex_[63][i] += wedge_count[ell] - 2;
                        orbit_counts.per_vertex_[63][ell] += wedge_count[ell] - 2;
                        orbit_counts.per_vertex_[64][j] += wedge_count[ell] - 2;
                        orbit_counts.per_vertex_[64][k] += wedge_count[ell] - 2;
                      }
                    }

                }

            }

            // looping over all diamonds for rdge (i, j), counting node orbits R_68 R_69
            for (EdgeIdx tri_pos = allTris->trioffsets[edge_pos]; tri_pos < allTris->trioffsets[edge_pos+1]; tri_pos++) // loop over triangles on edge (i, j)
            {
                k = allTris->triangles[tri_pos];

                VertexIdx degk = cg->offsets[k+1] - cg->offsets[k];
                VertexIdx new_pos = 0;
                if ((degk < degj) || (degk==degj && k <= j))
                    new_pos = cg->getEdgeBinary(k,j);
                else
                    new_pos = cg->getEdgeBinary(j,k);
            

                low = allTris->trioffsets[new_pos];
                high = allTris->trioffsets[new_pos+1]-1;

                while(low <= high)
                {
                     mid = (low+high)/2;
                     if (allTris->triangles[mid] == i)
                         break;
                     if (allTris->triangles[mid] > i)
                         high = mid-1;
                     else
                         low = mid+1;
                }

                for (EdgeIdx next_tri_pos = mid; next_tri_pos < allTris->trioffsets[new_pos+1]; next_tri_pos++) // loop over triangles on edge (j,k) with the third end ahead of i
                {
                    ell = allTris->triangles[next_tri_pos];

                    if(i < ell)
                    {
                        orbit_counts.per_vertex_[69][j] += choosec(diamond_count[ell], 2);
                        orbit_counts.per_vertex_[68][i] += choosec(diamond_count[ell], 2);
                        orbit_counts.per_vertex_[68][ell] += choosec(diamond_count[ell], 2);
                        orbit_counts.per_vertex_[68][k] += diamond_count_copy[ell] - 1;

                        diamond_count[ell] = 0;
                    }
                }
            }
        }

        
        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; ++pos) // loop over neighbors of i
        {
            j = cg->nbors[pos]; // j is current neighbor
            for (EdgeIdx next = cg->offsets[j]; next < cg->offsets[j+1]; ++next) // loop over neighbors of j
            {

                k = cg->nbors[next];  // i -- j -- k is wedge centered at j
                if (k <= i)
                    continue;
                orbit_counts.per_vertex_[62][j] += diamond_count_i[k];
            }
        } 

        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; ++pos) // loop over neighbors of i
        {
            j = cg->nbors[pos]; // j is current neighbor
            orbit_counts.per_vertex_[62][j] -= mid_diamond_count_i[j];
            mid_diamond_count_i[j] = 0;
            for (EdgeIdx next = cg->offsets[j]; next < cg->offsets[j+1]; ++next) // loop over neighbors of j
            {

                k = cg->nbors[next];  // i -- j -- k is wedge centered at j
                if (k <= i)
                    continue;
                wedge_count[k] = 0;
                diamond_count_i[k] = 0;
            }
        } 

   }

   // Node orbits R_68 and R_69 are double counted
   for (i=0; i < cg->nVertices; i++)
   {
        orbit_counts.per_vertex_[68][i] /= 2;
        orbit_counts.per_vertex_[69][i] /= 2;
   }

  delete [] wedge_count;
  delete [] diamond_count;
  delete [] diamond_count_copy;
  delete [] diamond_count_i;
  delete [] mid_diamond_count_i;

}


// This is a function that counts Node appearances in node orbit R_66
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG), a list of triangles for each edge, and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbit R_66

void Pattern66OrbitCounter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    VertexIdx *triends = new VertexIdx[gout->nVertices+1]; // array to store triangle ends
    EdgeIdx edge_pos_ij, edge_pos_ik, edge_pos_il, edge_pos_jk, edge_pos_jl, edge_pos_kl;

    EdgeIdx *edge_pos_i_triend = new VertexIdx[gout->nVertices+1]; // array to store triangle ends
    EdgeIdx *edge_pos_j_triend = new VertexIdx[gout->nVertices+1]; // array to store triangle ends

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

                        // edge_pos_ij = posj;
                        // edge_pos_ik = gout->getEdgeBinary(i, k);
                        // edge_pos_il = gout->getEdgeBinary(i, ell);
                        // edge_pos_jk = gout->getEdgeBinary(j, k);
                        // edge_pos_jl = gout->getEdgeBinary(j, ell);
                        edge_pos_kl = gout->getEdgeBinary(k, ell);

                        if (edge_pos_kl != -1) // (k,ell) is present, thus (i,j,k,ell) form a 4-clique
                        {
                            
                            orbit_counts.per_vertex_[66][i] += orbit_counts.per_edge_[1][edge_pos_jk]-2;
                            orbit_counts.per_vertex_[66][i] += orbit_counts.per_edge_[1][edge_pos_jl]-2;
                            orbit_counts.per_vertex_[66][i] += orbit_counts.per_edge_[1][edge_pos_kl]-2;

                            orbit_counts.per_vertex_[66][j] += orbit_counts.per_edge_[1][edge_pos_ik]-2;
                            orbit_counts.per_vertex_[66][j] += orbit_counts.per_edge_[1][edge_pos_il]-2;
                            orbit_counts.per_vertex_[66][j] += orbit_counts.per_edge_[1][edge_pos_kl]-2;

                            orbit_counts.per_vertex_[66][k] += orbit_counts.per_edge_[1][edge_pos_ij]-2;
                            orbit_counts.per_vertex_[66][k] += orbit_counts.per_edge_[1][edge_pos_il]-2;
                            orbit_counts.per_vertex_[66][k] += orbit_counts.per_edge_[1][edge_pos_jl]-2;

                            orbit_counts.per_vertex_[66][ell] += orbit_counts.per_edge_[1][edge_pos_ij]-2;
                            orbit_counts.per_vertex_[66][ell] += orbit_counts.per_edge_[1][edge_pos_ik]-2;
                            orbit_counts.per_vertex_[66][ell] += orbit_counts.per_edge_[1][edge_pos_jk]-2;
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

                        // edge_pos_ij = posj;
                        // edge_pos_ik = gout->getEdgeBinary(i, k);
                        // edge_pos_il = gout->getEdgeBinary(i, ell);
                        // edge_pos_jk = gout->getEdgeBinary(j, k);
                        // edge_pos_jl = gout->getEdgeBinary(j, ell);
                        // edge_pos_kl = gout->getEdgeBinary(k, ell);


                        if (binarySearch(triends+posk+1,count-posk-1,ell) != -1)
                        {
                            edge_pos_il = gout->getEdgeBinary(i, ell);
                            // edge_pos_jk = gout->getEdgeBinary(j, k);
                            edge_pos_jl = gout->getEdgeBinary(j, ell);

                            orbit_counts.per_vertex_[66][i] += orbit_counts.per_edge_[1][edge_pos_jk]-2;
                            orbit_counts.per_vertex_[66][i] += orbit_counts.per_edge_[1][edge_pos_jl]-2;
                            orbit_counts.per_vertex_[66][i] += orbit_counts.per_edge_[1][edge_pos_kl]-2;

                            orbit_counts.per_vertex_[66][j] += orbit_counts.per_edge_[1][edge_pos_ik]-2;
                            orbit_counts.per_vertex_[66][j] += orbit_counts.per_edge_[1][edge_pos_il]-2;
                            orbit_counts.per_vertex_[66][j] += orbit_counts.per_edge_[1][edge_pos_kl]-2;

                            orbit_counts.per_vertex_[66][k] += orbit_counts.per_edge_[1][edge_pos_ij]-2;
                            orbit_counts.per_vertex_[66][k] += orbit_counts.per_edge_[1][edge_pos_il]-2;
                            orbit_counts.per_vertex_[66][k] += orbit_counts.per_edge_[1][edge_pos_jl]-2;

                            orbit_counts.per_vertex_[66][ell] += orbit_counts.per_edge_[1][edge_pos_ij]-2;
                            orbit_counts.per_vertex_[66][ell] += orbit_counts.per_edge_[1][edge_pos_ik]-2;
                            orbit_counts.per_vertex_[66][ell] += orbit_counts.per_edge_[1][edge_pos_jk]-2;
                        }
                    }
                }
            }
        }
    }

  delete [] triends;
}


// This is a function that counts Node appearances in node orbits in the almost fivee clique pattern (R_70, R_71)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG), a list of triangles for each edge, and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbit in the almost fivee clique pattern (R_70, R_71)

void AlmostFiveCliqueOrbitCounter(CGraph *cg, CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    EdgeIdx *triangles = new VertexIdx[cg->nEdges+1];
    VertexIdx count = 0;

    for (VertexIdx i=0; i < cg->nVertices; i++) // loop over vertices
    {
        VertexIdx degi = cg->offsets[i+1] - cg->offsets[i]; // degree of node i
        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; pos++) // loop over neighbors of i
        {
            VertexIdx j = cg->nbors[pos]; // j is current neighbor of i
            VertexIdx degj = cg->offsets[j+1] - cg->offsets[j];
            
            if (degj < degi || (degj == degi && j < i))
                continue;
            count = 0;


            for (EdgeIdx ptr = cg->offsets[i]; ptr < cg->offsets[i+1]; ptr++) // loop over neighbors of i
            {
                VertexIdx nbr = cg->nbors[ptr]; // nbr is current neighbor
                
                if (cg->getEdgeBinary(nbr,j) != -1) // edge (nbr, j) is present
                {
                    triangles[count] = nbr; // adding nbr to the list of triangles for i and j
                    count++;
                }
            }

            std::sort(triangles,triangles+count);


            for (VertexIdx ptr = 0; ptr < count; ptr++) // loop over triangles on edge (i, j)
            {
                VertexIdx nbr = triangles[ptr]; // nbr is current node that makes a triangle with edge (i, j)
                VertexIdx degnbr = cg->offsets[nbr+1] - cg->offsets[nbr];

                if (degj < degnbr || (degj == degnbr && j < nbr))
                    continue;
                if (degnbr < degi || (degnbr == degi && nbr < i))
                    continue;

                VertexIdx fourclique = 0;
                
                if (count < degnbr)
                {
                    for (VertexIdx ptr2 = 0; ptr2 < count; ptr2++)
                    {
                        if (cg->getEdgeBinary(triangles[ptr2],nbr) != -1) // edge (triangle[ptr2], nbr) is present
                        {
                            // i,j,nbr,triangles[ptr2] is a 4-clique
                            fourclique++;
                        }
                    }
                }
                else
                {
                    for (VertexIdx next = cg->offsets[nbr]; next < cg->offsets[nbr+1]; next++)
                    {
                        if (binarySearch(triangles,count,cg->nbors[next]) != -1)
                        {
                            // i,j,cg->nbors[next], cg->nbors[next] is a 4-clique
                            fourclique++;
                        }
                    }
                }



                if (count < degnbr)
                {
                    for (VertexIdx ptr2 = 0; ptr2 < count; ptr2++)
                    {
                        if (cg->getEdgeBinary(triangles[ptr2],nbr) != -1)
                        {
                            //i, j, nbr, triangles[ptr2] is a 4-clique
                            orbit_counts.per_vertex_[70][triangles[ptr2]] += fourclique-1;
                        }
                    }
                }
                else
                {
                    for (VertexIdx next = cg->offsets[nbr]; next < cg->offsets[nbr+1]; next++)
                    {
                        if (binarySearch(triangles,count,cg->nbors[next]) != -1)
                        {
                            //i, j, nbr, cg->nbors[next] is a 4-clique
                            orbit_counts.per_vertex_[70][cg->nbors[next]] += fourclique-1;
                        }
                    }
                }

                orbit_counts.per_vertex_[71][i] += choosec(fourclique, 2);
                orbit_counts.per_vertex_[71][j] += choosec(fourclique, 2);
                orbit_counts.per_vertex_[71][nbr] += choosec(fourclique, 2);
            }
        }
    }

  delete [] triangles;
}


// This is a function that counts Node appearances in node orbit in the fivee clique pattern (R_72)
// Input: a pointer gout to a CGraph labeled according to degree (gout DAG), a pointer gin to a CGraph labeled according to degree (gin DAG), a list of triangles for each edge, and orbit_counts which is the structure in which we save the counts
// No Output: orbit_counts will have the orbit counts for orbit in the fivee clique pattern (R_72)

void FiveCliqueOrbitCounter(CGraph *gout, CGraph *gin, OrbitInfo &orbit_counts)
{
    VertexIdx *triends = new VertexIdx[gout->nVertices+1]; // array to store triangle ends
    VertexIdx *k4ends = new VertexIdx[gout->nVertices+1]; // array to store 4-clique ends

    for (VertexIdx i=0; i < gout->nVertices; ++i) // loop over vertices
        for (VertexIdx posj = gout->offsets[i]; posj < gout->offsets[i+1]; ++posj) // loop over out-neighbors of i
        {
            VertexIdx j = gout->nbors[posj]; // j is current out-neighbor
            VertexIdx count = 0;
            for (VertexIdx posk = posj+1; posk < gout->offsets[i+1]; ++posk) // loop over another out-neighbor of i, that is "ahead" of j in list of out-neighbors
            {
                VertexIdx k = gout->nbors[posk]; // k is next out-neighbor
                EdgeIdx edge_ind_jk = gout->getEdgeBinary(j,k);// check if edge (j,k) is present
                if (edge_ind_jk != -1) // edge (j,k) is present
                {
                    triends[count] = k;  // so (i,j,k) form a triangle. we store the fact that k forms a triangle with edge (i,j) in digraph gout
                    count++;
                }
            }

            for (VertexIdx posk = 0; posk < count; ++posk) // loop over all of triangles with edge (i,j)
            {
                VertexIdx count2 = 0;
                VertexIdx k = triends[posk]; // (i,j,k) is triangle
                for (VertexIdx posell = posk+1; posell < count; ++posell) // loop over another triangle with edge (i,j)
                {
                    VertexIdx ell = triends[posell]; // (i,j,ell) is next triangle
                    EdgeIdx edge_ind_kell = gout->getEdgeBinary(k,ell);
                    if (edge_ind_kell != -1) // (k,ell) is an present, thus (i,j,k,ell) form a 4-clique
                    {
                        k4ends[count2] = ell; // ell forms 4-clique with (i,j,k), so this is stored in k4ends
                        ++count2;
                    }
                }

                for (VertexIdx posell = 0; posell < count2; ++posell) // loop over all pairs of 4-cliques with triangle (i,j,k)
                    for (VertexIdx posoh = posell+1; posoh < count2; ++posoh)
                    {
                        VertexIdx ell = k4ends[posell]; // ell and oh are the vertices than form 4-cliques with triangle (i,j,k)
                        VertexIdx oh = k4ends[posoh];

                        if (gout->isEdge(ell,oh) != -1) // (ell,oh) is an edge, so (i,j,k,ell,oh) form 5-clique
                        {                         
                            orbit_counts.per_vertex_[72][i]++;
                            orbit_counts.per_vertex_[72][j]++;
                            orbit_counts.per_vertex_[72][k]++;
                            orbit_counts.per_vertex_[72][ell]++;
                            orbit_counts.per_vertex_[72][oh]++;
                        }
                    }
            }
        }

  delete [] triends;
  delete [] k4ends;
}


#endif



