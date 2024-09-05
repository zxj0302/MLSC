#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/Conversion.h"
#include <time.h>
#include <sys/time.h>

double Wtimediff (struct timeval start, struct timeval end)
{
  return ((double)end.tv_sec-start.tv_sec + ((double) (end.tv_usec-start.tv_usec))/1000000.0);
}

double Ctimediff (double start, double  end)
{
  return ((end-start)/CLOCKS_PER_SEC);
}

using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  struct timeval Wstart, Wend;
  double Cstart, Cend;

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

  printf("Counting the easier stuff\n");
  
  gettimeofday(&Wstart,NULL);
  Cstart=clock();
  SomeFourPatterns info = easyFourCounter(&cg_new, &(dag.outlist));
  gettimeofday(&Wend,NULL);
  Cend=clock();
  
  /* printf("3-stars = %ld\n",info.threestars);
  printf("3-paths = %ld\n",info.threepaths);
  printf("Tailed-triangles = %ld\n",info.tailedtris);
  printf("Chordal-cycles = %ld\n",info.chordalcycles); */
  printf(" Times (wall-clock, cpu time): %.4g %.4g\n", Wtimediff(Wstart,Wend), Ctimediff(Cstart,Cend)); 
  
  printf("Enumerating wedges for 4-cycles\n");
  gettimeofday(&Wstart,NULL);
  Cstart=clock();
  EdgeIdx fourcycles = fourCycleCounter(&(dag.outlist),&(dag.inlist));
  gettimeofday(&Wend,NULL);
  Cend=clock(); 
  /*  printf("Four-cycles = %ld\n",fourcycles); */
  printf(" Times (wall-clock, cpu time): %.4g %.4g\n", Wtimediff(Wstart,Wend), Ctimediff(Cstart,Cend));  
  printf("Enumerating 4-cliques\n"); 
//   (dag.outlist).print();

  gettimeofday(&Wstart,NULL);
  Cstart=clock();
  EdgeIdx fourcliques = fourCliqueCounter(&(dag.outlist));
  gettimeofday(&Wend,NULL);
  Cend=clock();
  /*  printf("Four-cliques= %ld\n",fourcliques); */
  printf(" Times (wall-clock, cpu time): %.4g %.4g\n", Wtimediff(Wstart,Wend), Ctimediff(Cstart,Cend)); 
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
