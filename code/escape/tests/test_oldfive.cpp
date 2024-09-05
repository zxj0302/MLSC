#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FourVertex.h"
#include "Escape/FiveTrees.h"
#include "Escape/FiveFromCycleClique.h"
#include "Escape/FiveFromTriangles.h"
#include "Escape/AlmostFiveClique.h"
#include "Escape/FiveCycle.h"
#include "Escape/WedgeCollisions.h"
#include "Escape/TriangleProgram.h"
#include "Escape/Utils.h"
#include "Escape/Conversion.h"

#include <iostream>
#include <fstream>
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
  CGraph cg_relabel = cg.renameByDegreeOrder();
  cg_relabel.sortById();
  printf("Creating DAG\n");
  CDAG dag = degreeOrdered(&cg_relabel);

  (dag.outlist).sortById();
  (dag.inlist).sortById();

//   (dag.outlist).print();
//   (dag.inlist).print();
  printf("Counting triangles\n");
  TriangleInfo tri_info = betterWedgeEnumerator(&(dag.outlist));

  printf("Getting all triangles\n");
  TriangleList allTris = storeAllTriangles(&cg_relabel,tri_info.total);

  printf("Also getting reverse triangle info\n");
  TriangleInfo in_tri_info = moveOutToIn(&(dag.outlist), &(dag.inlist), &tri_info);

//   for (VertexIdx i = 0; i < cg_relabel.nVertices; i++)
//       printf("%ld: %ld %ld\n\n",i,tri_info.perVertex[i],in_tri_info.perVertex[i]);
// 
//   for (VertexIdx i = 0; i < cg_relabel.nVertices; i++)
//       for (EdgeIdx posj = (dag.outlist).offsets[i]; posj < (dag.outlist).offsets[i+1]; posj++)
//       {
//           VertexIdx j = (dag.outlist).nbors[posj];
//           printf("%ld %ld: %ld\n",i,j,tri_info.perEdge[posj]);
//       }
// 
//   printf("\n");
//   for (VertexIdx i = 0; i < cg_relabel.nVertices; i++)
//       for (EdgeIdx posj = (dag.inlist).offsets[i]; posj < (dag.inlist).offsets[i+1]; posj++)
//       {
//           VertexIdx j = (dag.inlist).nbors[posj];
//           printf("%ld %ld: %ld\n",i,j,in_tri_info.perEdge[posj]);
//       }
// 
  printf("Counting the easier 4-vertex patterns\n");
  SomeFourPatterns four_info = easyFourCounter(&cg_relabel, &(dag.outlist));
  
  printf("Enumerating wedges for 4-cycles\n");
  EdgeIdx fourcycles = fourCycleCounter(&(dag.outlist),&(dag.inlist));
  
  printf("Enumerating 4-cliques\n"); 
  EdgeIdx fourcliques = fourCliqueCounter(&(dag.outlist));
  
  NoninducedFourCounts nonind;
  nonind.threestars = four_info.threestars;
  nonind.threepaths = four_info.threepaths;
  nonind.tailedtris = four_info.tailedtris;
  nonind.fourcycles = fourcycles;
  nonind.chordalcycles = four_info.chordalcycles;
  nonind.fourcliques = fourcliques;

  printf("Counting trees\n");
  FiveTrees tree_counts = fiveTreeCounter(&cg_relabel, nonind, tri_info.total);

  printf("Counting triangle based patterns\n");
  FiveFromTriangles tri_based_counts = fiveFromTriCounter(&cg_relabel,&(dag.outlist),&tri_info,four_info);

  Count hourglass = count5_Hourglass<false>(&dag.outlist, &tri_info, 0);
  Count stingray = count5_Stingray<false>(&dag.outlist, &dag.inlist, &tri_info, 0);
  Count three_tri_col = count5_StellateTrident<false>(&dag.outlist, &tri_info, 0);
  Count tri_strip = count5_TriangleStrip<true>(&dag.outlist, fourcliques, &tri_info);
  Count cobra = count5_Cobra<true>(&dag.outlist, &dag.inlist, fourcliques);
 
  printf("Counting 4-cycle and 4-clique based patterns\n");
  CycleBased cycle_related = fourCycleBasedCounter(&(dag.outlist), &(dag.inlist), &tri_info, &in_tri_info,nonind.chordalcycles);
  CliqueBased clique_related = fourCliqueBasedCounter(&cg_relabel, &(dag.outlist), &tri_info);

  printf("Counting five cycles\n");
  EdgeIdx five_cycle = fiveCycleCounter(&(dag.outlist),&(dag.inlist));

  printf("Counting collision patterns\n");
  CollisionPatterns collision_vals = fromTriangleList(&cg_relabel,&allTris);

  printf("Counting almost cliques\n");
  EdgeIdx almost_clique = almostFiveClique(&cg_relabel);  

  printf("--------------------\n");
  printf("Basic stats\n");
  printf("--------------------\n");
  printf("%ld vertices\n",cg_relabel.nVertices);
  printf("%ld edges\n",cg_relabel.nEdges);
  printf("%ld triangles\n",tri_info.total);
  
  printf("--------------------\n");
  printf("Non-induced 4-vertex counts\n");
  printf("--------------------\n");
  printf("%ld three-stars\n",nonind.threestars);
  printf("%ld three-paths\n",nonind.threepaths);
  printf("%ld tailed-tris\n",nonind.tailedtris);
  printf("%ld four-cycles\n",nonind.fourcycles);
  printf("%ld chordal-cycles\n",nonind.chordalcycles);
  printf("%ld 4-cliques\n",nonind.fourcliques);

  InducedFourCounts ind;
  ind = convertNonToInd(nonind);
  printf("--------------------\n");
  printf("Induced 4-vertex counts\n");
  printf("--------------------\n");
  printf("%ld three-stars\n",ind.threestars);
  printf("%ld three-paths\n",ind.threepaths);
  printf("%ld tailed-tris\n",ind.tailedtris);
  printf("%ld four-cycles\n",ind.fourcycles);
  printf("%ld chordal-cycles\n",ind.chordalcycles);
  printf("%ld 4-cliques\n",ind.fourcliques);

  double four_closures[6][6];
  
  closureMatrix(nonind,ind,four_closures);
  
  printf("--------------------\n");
  printf("4-vertex closure matrix\n");
  printf("--------------------\n");
   
  for (int i=0; i<6; i++)
  {
      for (int j=0; j<6; j++)
      {
          if (four_closures[i][j] > 0.005)
            printf("%.2f ",four_closures[i][j]);
          else
            printf("  x  ");
      }
      printf("\n");
  }

  EdgeIdx nonind_five[21], ind_five[21];

  nonind_five[0] = tree_counts.fourstars;
  nonind_five[1] = tree_counts.prongs;
  nonind_five[2] = tree_counts.fourpaths;
  nonind_five[3] = tri_based_counts.forktailedtris;
  nonind_five[4] = tri_based_counts.longtailedtris;
  nonind_five[5] = tri_based_counts.doubletailedtris;
  nonind_five[6] = cycle_related.tailedfourcycles;
  nonind_five[7] = five_cycle;
  nonind_five[8] = hourglass;
  nonind_five[9] =  cobra;
  nonind_five[10] =  stingray;
  nonind_five[11] = cycle_related.hattedfourcycles;
  nonind_five[12] = collision_vals.threeWedgeCol;
  nonind_five[13] = three_tri_col;
  nonind_five[14] = clique_related.tailedfourcliques;
  nonind_five[15] =  tri_strip;
  nonind_five[16] = collision_vals.chordalWedgeCol;
  nonind_five[17] = collision_vals.wheel;
  nonind_five[18] = clique_related.hattedfourcliques;
  nonind_five[19] = almost_clique;
  nonind_five[20] = clique_related.fivecliques;

  getFiveInduced(nonind_five, ind_five);

  printf("--------------------\n");
  printf("Non-induced 5-vertex counts\n");
  printf("--------------------\n");
  printf("%ld four-stars\n",nonind_five[0]);
  printf("%ld prongs\n",nonind_five[1]);
  printf("%ld four-paths\n",nonind_five[2]);
  printf("%ld forktailed-tris\n",nonind_five[3]);
  printf("%ld longtailed-tris\n",nonind_five[4]);
  printf("%ld doubletailed-tris\n",nonind_five[5]);
  printf("%ld tailed-4-cycle\n",nonind_five[6]);
  printf("%ld five-cycles\n",nonind_five[7]);
  printf("%ld hourglasses\n",nonind_five[8]);
  printf("%ld cobras \n",nonind_five[9]);
  printf("%ld stingrays\n",nonind_five[10]);
  printf("%ld hatted-4-cycle\n",nonind_five[11]);
  printf("%ld three-wedge collision\n",nonind_five[12]);
  printf("%ld three-triangle collision \n",nonind_five[13]);
  printf("%ld tailed-4-clique\n",nonind_five[14]);
  printf("%ld triangle strips \n",nonind_five[15]);
  printf("%ld chordal-cycle-wedge collision\n",nonind_five[16]);
  printf("%ld wheels\n",nonind_five[17]);
  printf("%ld hatted-4-clique\n",nonind_five[18]);
  printf("%ld almost-clique\n",nonind_five[19]);
  printf("%ld 5-cliques\n",nonind_five[20]);


  printf("--------------------\n");
  printf("Induced 5-vertex counts\n");
  printf("--------------------\n");
  printf("%ld four-stars\n",ind_five[0]);
  printf("%ld prongs\n",ind_five[1]);
  printf("%ld four-paths\n",ind_five[2]);
  printf("%ld forktailed-tris\n",ind_five[3]);
  printf("%ld longtailed-tris\n",ind_five[4]);
  printf("%ld doubletailed-tris\n",ind_five[5]);
  printf("%ld tailed-4-cycle\n",ind_five[6]);
  printf("%ld five-cycles\n",ind_five[7]);
  printf("%ld hourglasses\n",ind_five[8]);
  printf("%ld cobras \n",ind_five[9]);
  printf("%ld stingrays\n",ind_five[10]);
  printf("%ld hatted-4-cycle\n",ind_five[11]);
  printf("%ld three-wedge collision\n",ind_five[12]);
  printf("%ld three-triangle collision \n",ind_five[13]);
  printf("%ld tailed-4-clique\n",ind_five[14]);
  printf("%ld triangle strips \n",ind_five[15]);
  printf("%ld chordal-cycle-wedge collision\n",ind_five[16]);
  printf("%ld wheels\n",ind_five[17]);
  printf("%ld hatted-4-clique\n",ind_five[18]);
  printf("%ld almost-clique\n",ind_five[19]);
  printf("%ld 5-cliques\n",ind_five[20]);
  

  double closures_five[21][21];
  
  closureMatrixFive(nonind_five,ind_five,closures_five);
  
  printf("--------------------\n");
  printf("5-vertex closure matrix\n");
  printf("--------------------\n");
   
  printf("    ");
  for (int i=0; i<9; i++)
      printf("  %d  ",i+1);
  for (int i=9; i<21; i++)
      printf(" %d  ",i+1);

  printf("\n");

  for (int i=0; i<21; i++)
      printf("-----");

  printf("\n");

  for (int i=0; i<21; i++)
  {
      if (i<9)
          printf(" %d: ",i+1);
      else
          printf("%d: ",i+1);
      for (int j=0; j<21; j++)
      {
          if (closures_five[i][j] > 0.005)
            printf("%.2f ",closures_five[i][j]);
          else
            printf("  x  ");
      }
      printf("|  %d\n",i+1);
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  std::ofstream out_file;
  out_file.open("nums.txt");
  out_file << nonind.threestars;
  out_file << "\n";
  out_file << nonind.threepaths;
  out_file << "\n";
  out_file << nonind.tailedtris;
  out_file << "\n";
  out_file << nonind.fourcycles;
  out_file << "\n";
  out_file << nonind.chordalcycles;
  out_file << "\n";
  out_file << fourcliques;
  out_file << "\n";
  out_file << tree_counts.fourstars;
  out_file << "\n";
  out_file << tree_counts.prongs;
  out_file << "\n";
  out_file << tree_counts.fourpaths;
  out_file << "\n";
  out_file << tri_based_counts.forktailedtris;
  out_file << "\n";
  out_file << tri_based_counts.longtailedtris;
  out_file << "\n";
  out_file << tri_based_counts.doubletailedtris;
  out_file << "\n";
  out_file << cycle_related.tailedfourcycles;
  out_file << "\n";
  out_file << five_cycle;
  out_file << "\n";
  out_file << hourglass;
  out_file << "\n";
  out_file <<  cobra;
  out_file << "\n";
  out_file <<  stingray;
  out_file << "\n";
  out_file << cycle_related.hattedfourcycles;
  out_file << "\n";
  out_file << collision_vals.threeWedgeCol;
  out_file << "\n";
  out_file << three_tri_col;
  out_file << "\n";
  out_file << clique_related.tailedfourcliques;;
  out_file << "\n";
  out_file <<  tri_strip;;
  out_file << "\n";
  out_file << collision_vals.chordalWedgeCol;
  out_file << "\n";
  out_file << collision_vals.wheel;
  out_file << "\n";
  out_file << clique_related.hattedfourcliques;
  out_file << "\n";
  out_file << almost_clique;
  out_file << "\n";
  out_file << clique_related.fivecliques;
  out_file << "\n";
}

