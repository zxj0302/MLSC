#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/Graph.h"
#include "Escape/GetAllCounts.h"
#include "Escape/GetAllOrbitCounts.h"
#include "Escape/OrbitStructure.h"
#include <iostream>
#include <string.h>

using namespace std;
using namespace Escape;

// We want to count node orbits and edge orbits in patters of size 4, 3, 2, and 1.

int main(int argc, char *argv[])
{
  int pattern_size = 4;
  int num_orbits = 15; // number of different node orbits in all the patterns
  int num_edge_orbits = 12; // number of different edge orbits in all the patterns

  bool node_only = false;
  bool orbit_rows_vertex_cols = false;
  if (argc > 2)
  {
    if(strcmp(argv[2], "node") == 0)
    {
      node_only = true;
      if(argc > 3 && strcmp(argv[3], "vr") == 0)
      {
        orbit_rows_vertex_cols = true;
      }
    }
    else if(strcmp(argv[2], "vr") == 0)
    {
      orbit_rows_vertex_cols = true;
    }
  }

  Graph g;
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Loaded graph\n");
  CGraph cg = makeCSR(g);	
  cg.sortById();
  printf("Converted to CSR\n");

  printf("Relabeling graph\n");
  VertexIdx *mapping; // mapping[v] stores the node mapped to vertex v.
  VertexIdx *inverse; // inverse[v] stores the node to which v is mapped.
  CGraph cg_relabel = cg.renameByDegreeOrder(mapping, inverse);
  cg_relabel.sortById();
  printf("Creating DAG\n");
  CDAG dag = degreeOrdered(&cg_relabel);

  (dag.outlist).sortById();
  (dag.inlist).sortById();


  /*
  *OLD OLD OLD
  *

  // OrbitInfo orbit_counts(&(dag.outlist), num_node_patterns, num_edge_patterns);
  
  *
  *OLD OLD OLD
  */
  

  // Creating a structure to save the orbit counts information
  OrbitInfo orbit_counts(&(dag.outlist), num_orbits, num_edge_orbits);

  // Getting all the counts
  get_all_three_orbits(&cg_relabel, &dag, orbit_counts);
  if (node_only)
    get_all_four_orbits_node_only(&cg_relabel, &dag, orbit_counts);
  else
    get_all_four_orbits(&cg_relabel, &dag, orbit_counts);

  // Printing the output to out.txt file
  FILE* f = fopen("out.txt","w");
  if (!f)
  {
      printf("could not write to output to out.txt\n");
      return 0;
  }
  if(orbit_rows_vertex_cols == false)
  {
    for(int i = 0; i < cg_relabel.nVertices; i++)
    {
        for(int j=0; j < num_orbits; j++)
        {
          fprintf(f,"%lf ", orbit_counts.per_vertex_[j][mapping[i]]);
        }
        fprintf(f,"\t\n");
    } 
  }
  else
  {
    for(int i = 0; i < num_orbits; i++)
    {
        for(int j=0; j < cg_relabel.nVertices; j++)
        {
          fprintf(f,"%lf ", orbit_counts.per_vertex_[i][mapping[j]]);
        }
        fprintf(f,"\t\n");
    } 
  }
  fclose(f);  
  
  delete [] mapping;
  delete [] inverse;
}

