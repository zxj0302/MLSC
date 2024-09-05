#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/Conversion.h"
#include "Escape/GetAllCounts.h"


using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Loaded graph\n");
  CGraph cg = makeCSR(g);
  cg.sortById();
  printf("Converted to CSR\n");

  printf("Relabeling graph\n");
  CGraph cg_relabel = cg.renameByDegreeOrder();
  cg_relabel.sortById();
  printf("Creating DAG\n");
  CDAG dag = degreeOrdered(&cg_relabel);

  (dag.outlist).sortById();
  (dag.inlist).sortById();

  printf("Counting 4-cliques\n");
  EdgeIdx fourcliques = fourCliqueCounter(&(dag.outlist));

  printf("Original data\n");
  printf("----------------------\n");
  printf("%ld edges\n",cg_relabel.nEdges);
  printf("%ld 4-cliques\n",fourcliques);
  printf("----------------------\n");

  printf("Counting triangles\n");
  TriangleInfo tri_info = betterWedgeEnumerator(&(dag.outlist));

  printf("Getting all triangles\n");
  TriangleList allTris = storeAllTriangles(&cg_relabel,tri_info.total);

  printf("Computing 1 truss\n");
  CGraph truss = cTruss(&cg_relabel,&allTris,1); 
  CGraph truss_relabel = truss.renameByDegreeOrder();
  truss_relabel.sortById();
  CDAG truss_dag = degreeOrdered(&truss_relabel);
  EdgeIdx trussEdges1 = truss_relabel.nEdges;

  printf("Counting 4-cliques in 1-truss\n");
  EdgeIdx fourcliques1 = fourCliqueCounter(&(truss_dag.outlist));

  printf("1-truss data\n");
  printf("----------------------\n");
  printf("%ld edges\n",trussEdges1);
  printf("%ld 4-cliques\n",fourcliques1);
  printf("----------------------\n");

  printf("Computing 2 truss\n");
  truss = cTruss(&cg_relabel,&allTris,2); 
  truss_relabel = truss.renameByDegreeOrder();
  truss_relabel.sortById();
  truss_dag = degreeOrdered(&truss_relabel);
  EdgeIdx trussEdges2 = truss_relabel.nEdges;

  printf("Counting 4-cliques in 2-truss\n");
  EdgeIdx fourcliques2 = fourCliqueCounter(&(truss_dag.outlist));

  printf("2-truss data\n");
  printf("----------------------\n");
  printf("%ld edges\n",trussEdges2);
  printf("%ld 4-cliques\n",fourcliques2);
  printf("----------------------\n");

}

