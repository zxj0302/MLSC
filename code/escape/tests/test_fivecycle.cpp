#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/FiveFromCycleClique.h"
#include "Escape/FiveCycle.h"

using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  printf("Loading graph\n");
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Converting to CSR\n");
  CGraph cg = makeCSR(g);

   printf("Creating DAG\n");
   CDAG dag = degreeOrdered(&cg);
   
//   printf("Counting the easier 4-vertex patterns\n");
//   SomeFourPatterns info = easyFourCounter(&cg, &(dag.outlist));
// 
//    (dag.outlist).print();
//    (dag.inlist).print();
 
   printf("Getting five cycle count\n");
   EdgeIdx count = fiveCycleCounter(&(dag.outlist),&(dag.inlist));

//    printf("Tailed tris = %ld\n",info.tailedtris);
   printf("Five cycles = %ld\n",count);
}
