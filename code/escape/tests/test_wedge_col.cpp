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

  printf("Creating DAG\n");
  CDAG dag = degreeOrdered(&cg);

  printf("Counting triangles\n");
  TriangleInfo tri_count = wedgeEnumerator(&(dag.outlist));

  printf("Getting all triangles\n");
  TriangleList allTris = storeAllTriangles(&cg,tri_count.total);
//   
//   EdgeIdx count = threeWedgeFromFourCycle(&(dag.outlist), &(dag.inlist));
//       printf("Getting old three-way wedge collision\n");
//       EdgeIdx count = threeWedgeCollision(&cg);

  printf("Newer collision counter\n");
  CollisionPatterns val = fromTriangleList(&cg,&allTris);
//   CollisionPatterns val = oldWedgeBasedCollision(&cg,&(dag.outlist));
//   CollisionPatterns val = newWedgeBasedCollision(&cg,&(dag.outlist));


  printf("Three-way wedge collision = %ld\n",val.threeWedgeCol);
  printf("Chordal-cycle-wedge collision = %ld\n",val.chordalWedgeCol);
  printf("Wheels = %ld\n",val.wheel);
    

  if (argc > 2)
  {
    
      printf("Getting old chordal-cycle-wedge collision\n");
      CollisionPatterns ccw = chordalCycleWedgeCollision(&cg,&(dag.outlist));

      printf("Old Three-way wedge collision = %ld\n",ccw.threeWedgeCol);
      printf("Old Chordal-cycle-wedge collision = %ld\n",ccw.chordalWedgeCol);
  }

}
