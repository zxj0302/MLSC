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

  printf("Counting triangles\n");
  TriangleInfo tri_info = betterWedgeEnumerator(&(dag.outlist));

  printf("Getting all triangles\n");
  TriangleList allTris = storeAllTriangles(&cg_relabel,tri_info.total);

  printf("Computing 2 truss\n");
  CGraph truss = cTruss(&cg_relabel,&allTris,2); 
  CGraph truss_relabel = truss.renameByDegreeOrder();
  truss_relabel.sortById();
  CDAG truss_dag = degreeOrdered(&truss_relabel);
  EdgeIdx trussEdges2 = truss_relabel.nEdges;

  printf("Counting almost-cliques in 2-truss\n");
  EdgeIdx count = almostFiveClique(&cg_relabel);

  printf("Almost five cliques = %ld\n",count);

}

