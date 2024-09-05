#include "Escape/GraphIO.h"
#include "Escape/TriangleProgram.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"


using namespace Escape;


struct TriangleCounter
{
  size_t count = 0;

  void operator ()(VertexIdx u, VertexIdx v, VertexIdx w, EdgeIdx vu, EdgeIdx vw, EdgeIdx uw)
  {
    ++count;
  }
};



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
  
  auto tc1 = TriangleCounter();  
  triangleProgram<true>(&(dag.outlist), tc1);

  auto tc2 = TriangleCounter();
  triangleProgram<false, true, true>(&cg, tc2);
  printf("Reference: %ld, with doDegreeOrdering: %ld, without: %ld\n", info.total, tc1.count, tc2.count);
}