#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/Graph.h"

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

  CDAG dag = degreeOrdered(&cg);
  printf("Created DAG\n");

  TriangleInfo info;
  if (argc > 2)
  {
      printf("Old version\n");
      info = wedgeEnumerator(&(dag.outlist));
      printf("Enumerated wedges\n");
      printf("Triangles = %ld\n",info.total);
  }

  printf("New version\n");
  printf("Relabeling graph\n");
  CGraph cg_new = cg.renameByDegreeOrder();
  cg_new.sortById();
  
  dag = degreeOrdered(&cg_new);


  info = betterWedgeEnumerator(&(dag.outlist));
  printf("Enumerated wedges\n");
  printf("Triangles = %ld\n",info.total);

  printf("dag has %ld edges\n", dag.inlist.nEdges);
}
