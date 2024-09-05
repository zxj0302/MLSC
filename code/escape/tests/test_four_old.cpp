#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/Conversion.h"

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
  
  if (argc > 2)
  {
      printf("Older code\n\n");
    
      printf("Counting the easier stuff\n");
      SomeFourPatterns info = oldeasyFourCounter(&cg, &(dag.outlist));
      printf("3-stars = %ld\n",info.threestars);
      printf("3-paths = %ld\n",info.threepaths);
      printf("Tailed-triangles = %ld\n",info.tailedtris);
      printf("Chordal-cycles = %ld\n",info.chordalcycles);

      
      printf("Enumerating wedges for 4-cycles\n");
      EdgeIdx fourcycles = oldfourCycleCounter(&(dag.outlist),&(dag.inlist));
      printf("Four-cycles = %ld\n",fourcycles);
      
      printf("Enumerating 4-cliques\n"); 
      EdgeIdx fourcliques = oldfourCliqueCounter(&(dag.outlist));
      printf("Four-cliques= %ld\n",fourcliques);
  }

  printf("\n\nNewer code\n\n");
  printf("Relabeling graph\n");
  CGraph cg_new = cg.renameByDegreeOrder();
  cg_new.sortById();
  printf("Creating DAG\n");
  dag = degreeOrdered(&cg_new);

  printf("Counting the easier stuff\n");
  SomeFourPatterns info = easyFourCounter(&cg_new, &(dag.outlist));
  printf("3-stars = %ld\n",info.threestars);
  printf("3-paths = %ld\n",info.threepaths);
  printf("Tailed-triangles = %ld\n",info.tailedtris);
  printf("Chordal-cycles = %ld\n",info.chordalcycles);

  
  printf("Enumerating wedges for 4-cycles\n");
  EdgeIdx fourcycles = fourCycleCounter(&(dag.outlist),&(dag.inlist));
  printf("Four-cycles = %ld\n",fourcycles);
  
  printf("Enumerating 4-cliques\n"); 
//   (dag.outlist).print();

  EdgeIdx fourcliques = fourCliqueCounter(&(dag.outlist));
  printf("Four-cliques= %ld\n",fourcliques);

  NoninducedFourCounts nonind;
  InducedFourCounts ind;

  nonind.threestars = info.threestars;
  nonind.threepaths = info.threepaths;
  nonind.tailedtris = info.tailedtris;
  nonind.fourcycles = fourcycles;
  nonind.chordalcycles = info.chordalcycles;
  nonind.fourcliques = fourcliques;

  ind = convertNonToInd(nonind);

  printf("\nNon-induced counts:\n");

  printf("3-stars = %ld\n",nonind.threestars);
  printf("3-paths = %ld\n",nonind.threepaths);
  printf("Tailed-triangles = %ld\n",nonind.tailedtris);
  printf("Four-cycles = %ld\n",nonind.fourcycles);
  printf("Chordal-cycles = %ld\n",nonind.chordalcycles);
  printf("Four-cliques= %ld\n",nonind.fourcliques);

  printf("\nInduced counts:\n");

  printf("3-stars = %ld\n",ind.threestars);
  printf("3-paths = %ld\n",ind.threepaths);
  printf("Tailed-triangles = %ld\n",ind.tailedtris);
  printf("Four-cycles = %ld\n",ind.fourcycles);
  printf("Chordal-cycles = %ld\n",ind.chordalcycles);
  printf("Four-cliques= %ld\n",ind.fourcliques);

}
