#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/Conversion.h"
#include "Escape/GetAllCounts.h"


using namespace Escape;

int main(int argc, char *argv[])
{
  Graph g;
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);

  printf("Loaded graph\n");
  CGraph cg = makeCSR(g);
  cg.sortById();
  printf("Converted to CSR\n");

  printf("Relabeling graph\n");
  CGraph cg_relabel = cg.renameByDegreeOrder();
  cg_relabel.sortById();
  printf("Creating DAG\n");
  CDAG dag = degreeOrdered(&cg_relabel);

  printf("Counting the easier stuff\n");
  
  SomeFourPatterns info = easyFourCounter(&cg_relabel, &(dag.outlist));
  
  printf("Enumerating wedges for 4-cycles\n");
  EdgeIdx fourcycles = fourCycleCounter(&(dag.outlist),&(dag.inlist));
  
  printf("Counting triangles\n");
  TriangleInfo tri_info = betterWedgeEnumerator(&(dag.outlist));

  printf("Getting all triangles\n");
  TriangleList allTris = storeAllTriangles(&cg_relabel,tri_info.total);

  printf("Computing 2 truss\n");
  CGraph truss = cTruss(&cg_relabel,&allTris,2); 
  CGraph truss_relabel = truss.renameByDegreeOrder();
  truss_relabel.sortById();
  CDAG truss_dag = degreeOrdered(&truss_relabel);
  EdgeIdx trussEdges2 = truss_relabel.nEdges;

  printf("Counting 4-cliques in 2-truss\n");
  EdgeIdx fourcliques = fourCliqueCounter(&(truss_dag.outlist));

  NoninducedFourCounts nonind;
  InducedFourCounts ind;

  nonind.threestars = info.threestars;
  nonind.threepaths = info.threepaths;
  nonind.tailedtris = info.tailedtris;
  nonind.fourcycles = fourcycles; 
  nonind.chordalcycles = info.chordalcycles;
  nonind.fourcliques = fourcliques;

  ind = convertNonToInd(nonind);

  printf("%ld vertices, %ld edges, %ld 2-truss edges\n",cg_relabel.nVertices, cg_relabel.nEdges, trussEdges2);

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

