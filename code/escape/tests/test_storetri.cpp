#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"

using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Loaded graph\n");
  CGraph cg = makeCSR(g);
  printf("Converted to CSR\n");

  CDAG dag = degreeOrdered(&cg);
  printf("Created DAG\n");
  TriangleInfo info = wedgeEnumerator(&(dag.outlist));
  printf("Enumerated wedges\n");
  printf("Triangles = %ld\n",info.total);

  TriangleList everything = storeAllTriangles(&cg,info.total);
}
