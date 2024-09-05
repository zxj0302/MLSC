#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/FiveFromCycleClique.h"
#include "Escape/WedgeCollisions.h"

using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  printf("Loading graph\n");
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Converting to CSR\n");
  CGraph cg = makeCSR(g);

//   printf("Getting square data\n");
//   EdgeIdx count = squareDetails(&cg);

  printf("Generating square\n");
  generateSquare(&cg,10);



}
