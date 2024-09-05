#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"


using namespace Escape;


int main(int argc, char *argv[])
{
  Graph g;
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);
  g.print();

  CGraph cg = makeCSR(g);
  cg.print();

  CGraph cg_new = cg.renameByDegreeOrder();
  cg_new.print();

  delGraph(g);
  delCGraph(cg);
  delCGraph(cg_new);
}
