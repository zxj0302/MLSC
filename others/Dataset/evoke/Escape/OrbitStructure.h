#ifndef ESCAPE_ORBITSTRUCTURE_H_
#define ESCAPE_ORBITSTRUCTURE_H_

#include "Escape/ErrorCode.h"
#include "Escape/Graph.h"
#include "Escape/Digraph.h"

using namespace Escape;

// structure that stores the orbits counts (number of times each node appears in each orbit).
struct OrbitInfo
{
  int num_orbits_;
  int num_edge_orbits_;
  double **per_vertex_; // per_vertex[i][v] stores R_i(v)
  double **per_edge_; // per_edge[i][e] stores E_i(e) where e is the edge index in gout DAG
  double **per_edge_rev_; // in the case that the two nodes at the end of the edge in the edge orbits are in different node orbits, we give the edge a direction and per_edge saves the case that the edge orbit is in the same diretion with gout DAG. In this case per_edge_rev_ saves the other case. Otherwise it is 0.
  
  OrbitInfo(CGraph *graph, const int num_orbits, const int num_edge_orbits);
  ~OrbitInfo();
};

// allocates memory and initializes everything to zero
OrbitInfo::OrbitInfo(CGraph *graph, const int num_orbits, const int num_edge_orbits)
{
  num_orbits_ = num_orbits;
  num_edge_orbits_ = num_edge_orbits;

  per_vertex_ = new double*[num_orbits+1];
  for(int i=0; i < num_orbits; i++)
  {
    per_vertex_[i] = new double[graph->nVertices];
    for(int j=0; j < graph->nVertices; j++){
      per_vertex_[i][j] = 0;
    }
  }

  per_edge_ = new double*[num_edge_orbits+1];
  per_edge_rev_ = new double*[num_edge_orbits+1];
  for(int i=0; i < num_edge_orbits; i++)
  {
    per_edge_[i] = new double[graph->nEdges+1];
    per_edge_rev_[i] = new double[graph->nEdges+1];
    for(int j=0; j < graph->nEdges; j++){
      per_edge_[i][j] = 0;
      per_edge_rev_[i][j] = 0;
    }
  }
}

OrbitInfo::~OrbitInfo()
{
  for(int i=0; i < num_orbits_; i++)
  {
    delete [] per_vertex_[i];
  }
  delete [] per_vertex_;

  for(int i=0; i < num_edge_orbits_; i++)
  {
    delete [] per_edge_[i];
    delete [] per_edge_rev_[i];
  }
  delete [] per_edge_;
  delete [] per_edge_rev_;
}


/*
* OLD OLD OLD
*

// //template <class T>
// OrbitInfo* newOrbitInfo(CGraph *graph, const int num_node_patterns, const int num_edge_patterns)
// {
//   OrbitInfo *ret;
  
//   ret -> per_vertex_ = new Count*[num_node_patterns+1];
//   for(int i=0; i < num_node_patterns; i++)
//   {
//   	ret -> per_vertex_[i] = new Count[graph->nVertices];
//   }

//   ret -> per_edge_ = new Count*[num_edge_patterns+1];
//   for(int i=0; i < num_edge_patterns; i++)
//   {
//   	ret -> per_edge_[i] = new Count[graph->nEdges+1];
//   }

//   ret -> per_edge_rev_ = new Count*[num_edge_patterns+1];
//   for(int i=0; i < num_edge_patterns; i++)
//   {
//   	ret -> per_edge_rev_[i] = new Count[graph->nEdges+1];
//   }
  
//   return ret;
// }



// //template <class T>
// OrbitInfo newOrbitInfo(CGraph *graph, const int num_node_patterns, const int num_edge_patterns)
// {
//   OrbitInfo ret;
  
//   ret.per_vertex_ = new Count*[num_node_patterns+1];
//   for(int i=0; i < num_node_patterns; i++)
//   {
//     ret.per_vertex_[i] = new Count[graph->nVertices];
//   }

//   ret.per_edge_ = new Count*[num_edge_patterns+1];
//   for(int i=0; i < num_edge_patterns; i++)
//   {
//     ret.per_edge_[i] = new Count[graph->nEdges+1];
//   }

//   ret.per_edge_rev_ = new Count*[num_edge_patterns+1];
//   for(int i=0; i < num_edge_patterns; i++)
//   {
//     ret.per_edge_rev_[i] = new Count[graph->nEdges+1];
//   }
  
//   return ret;
// }


*
* OLD OLD OLD
*/ 


#endif

