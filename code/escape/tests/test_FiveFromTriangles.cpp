#include "Escape/GraphIO.h"
#include "Escape/TriangleProgram.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/FiveFromTriangles.h"


using namespace Escape;


int main(int argc, const char *argv[])
{
  Graph g;
  if (loadGraph(argv[1], g, 1, IOFormat::escape))
    exit(1);
  printf("Loaded graph\n");
  CGraph gOut = makeCSR(g);
  CGraph gIn  = makeCSC(g);

  printf("Relabeling graph\n");
  CGraph cg_new = gOut.renameByDegreeOrder();
  cg_new.sortById();

  CDAG dag = degreeOrdered(&cg_new);
  printf("Created DAG\n");
  TriangleInfo tInfo = betterWedgeEnumerator(&(dag.outlist));
  printf("Counting the easier four vertex stuff\n");
  SomeFourPatterns easycounts = easyFourCounter(&cg_new, &(dag.outlist));
  printf("Enumerating 4-cliques\n"); 
  EdgeIdx fourcliques = fourCliqueCounter(&(dag.outlist));

  printf("%ld =? %ld triangles\n", tInfo.total, tInfo.total);
  printf("%ld tailed-tris\n",easycounts.tailedtris);
  printf("%ld chordal-cycles\n",easycounts.chordalcycles);
  printf("%ld 4-cliques\n",fourcliques);

  FiveFromTriangles counts = fiveFromTriCounter(&cg_new,&(dag.outlist),&tInfo,easycounts);

  printf("%ld forktailed-tris\n",counts.forktailedtris);
  printf("%ld longtailed-tris\n",counts.longtailedtris);
  printf("%ld doubletailed-tris\n",counts.doubletailedtris);
  
//  auto tInfo = newTriangleInfo(&gOut);
//  tInfo.total = countTriangles(&gOut, false, true, &gIn, tInfo.perVertex, tInfo.perEdge);
  printf("%ld triangles\n", tInfo.total);
//   for (VertexIdx v = 0; v < dag.outlist.nVertices; ++v)
//     printf("%ld %ld\n", v, tInfo.perVertex[v]);
//   for (EdgeIdx e = 0; e < dag.outlist.nEdges; ++e)
//     printf("%ld %ld\n", e, tInfo.perEdge[e]);

  Count n = count5_Hourglass<false>(&dag.outlist, &tInfo, 0);
  printf("%ld hourglasses\n", n);

  n = count5_Stingray<false>(&dag.outlist, &dag.inlist, &tInfo, 0);
  printf("%ld stingrays\n", n);

  n = count5_StellateTrident<false>(&dag.outlist, &tInfo, 0);
  printf("%ld stellate tridents\n", n);

  //this will be incorrect, since we're setting the 4-cliques to zero here!!
  n = count5_TriangleStrip<true>(&dag.outlist, fourcliques, &tInfo);
  printf("%ld triangle strips \n", n);

  //this will be incorrect since we're passing in total 4-cliques as zero.
  n = count5_Cobra<true>(&dag.outlist, &dag.inlist, fourcliques);
  printf("%ld cobras \n", n);
  
  delTriangleInfo(tInfo);
  delCGraph(dag.inlist);
  delCGraph(dag.outlist);
  delGraph(g);
  delCGraph(gOut);
  delCGraph(gIn);
}
