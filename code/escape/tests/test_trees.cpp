#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FiveTrees.h"
#include "Escape/FourVertex.h"

using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  printf("Loading graph\n");
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Converting to CSR\n");
  CGraph cg = makeCSR(g);

  printf("Relabeling graph\n");
  CGraph cg_new = cg.renameByDegreeOrder();
  cg_new.sortById();
  printf("Creating DAG\n");
  CDAG dag = degreeOrdered(&cg_new);

  printf("Counting triangles\n");
  TriangleInfo tri_info = betterWedgeEnumerator(&(dag.outlist));

  printf("Counting the easier stuff\n");
  SomeFourPatterns info = easyFourCounter(&cg_new, &(dag.outlist));
  
  printf("Enumerating wedges for 4-cycles\n");
  EdgeIdx fourcycles = fourCycleCounter(&(dag.outlist),&(dag.inlist));
  
  NoninducedFourCounts nonind;

  nonind.threestars = info.threestars;
  nonind.threepaths = info.threepaths;
  nonind.tailedtris = info.tailedtris;
  nonind.fourcycles = fourcycles;
  nonind.chordalcycles = info.chordalcycles;
  nonind.fourcliques = 0;

  printf("Tailed tris = %ld\n",info.tailedtris);
  printf("Four cycles = %ld\n",fourcycles);

  FiveTrees counts = fiveTreeCounter(&cg_new, nonind, tri_info.total);
  printf("Four-stars = %ld\n",counts.fourstars);
  printf("Prongs = %ld\n",counts.prongs);
  printf("Four-paths = %ld\n",counts.fourpaths);
}
