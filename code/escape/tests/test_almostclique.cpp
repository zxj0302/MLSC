#include "Escape/GraphIO.h"
#include "Escape/Graph.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/FiveFromCycleClique.h"
#include "Escape/AlmostFiveClique.h"

using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  printf("Loading graph\n");
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Converting to CSR\n");
  CGraph cg = makeCSR(g);

   printf("Sorting by ID\n");
   cg.sortById();

   printf("Getting almost 5-clique count\n");
   EdgeIdx count = almostFiveClique(&cg);

   printf("Almost five cliques from new = %ld\n",count);

   if (argc > 2)
   {
       printf("Running old code\n");
       EdgeIdx count2 = oldAlmostFiveClique(&cg);
       printf("Almost five cliques from old = %ld\n",count2);
   }
}
