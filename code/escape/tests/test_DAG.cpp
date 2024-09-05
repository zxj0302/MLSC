#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"

using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);
  g.print();

  CGraph cg = makeCSR(g);
  cg.print();

  CDAG dag = degreeOrdered(&cg);
  printf("safe return\n");
  CGraph outdag = dag.outlist;
  CGraph indag = dag.inlist;

  (outdag).print();
  (indag).print();

  printf("New version\n");
  printf("Relabeling graph\n");
  CGraph cg_new = cg.renameByDegreeOrder();
  cg_new.sortById();
  
  dag = degreeOrdered(&cg_new);

  printf("safe return\n");
  outdag = dag.outlist;
  indag = dag.inlist;

  (outdag).print();
  (indag).print();


  delGraph(g);
  delCGraph(cg);
  delCGraph(outdag);
  delCGraph(indag);
}
