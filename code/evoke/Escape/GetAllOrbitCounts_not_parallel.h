#ifndef ESCAPE_GETALLORBITCOUNTS_H_
#define ESCAPE_GETALLORBITCOUNTS_H_

#include <algorithm>

#include "Escape/FourVertex.h"
#include "Escape/Utils.h"
#include "Escape/GraphIO.h"
#include "Escape/EdgeHash.h"
#include "Escape/Digraph.h"
#include "Escape/Triadic.h"
#include "Escape/Graph.h"
#include "Escape/Conversion.h"
#include "Escape/FiveTrees.h"
#include "Escape/FiveFromCycleClique.h"
#include "Escape/FiveFromTriangles.h"
#include "Escape/AlmostFiveClique.h"
#include "Escape/FiveCycle.h"
#include "Escape/WedgeCollisions.h"
#include "Escape/TriangleProgram.h"
#include "Escape/OrbitStructure.h"
#include "Escape/ThreeVertexOrbit.h"
#include "Escape/FourVertexOrbit.h"
#include "Escape/FiveVertexOrbit.h"

#include <iostream>
#include <iomanip>
#include <string>

using namespace std;
using namespace Escape;


// This function generates all orbit counts and edge orbit counts for orbits and edge orbits in patterns with 3, 2, and 1 vertex.
// It is a wrapper function that calls the main algorithmic parts

// Input: pointer to CGraph, corresponding DAG, OrbitInfo structure to store orbit counts
// No output: orbit_counts will have all the orbit and edge orbit counts

void get_all_three_orbits(CGraph *cg, CDAG *dag, OrbitInfo &orbit_counts)
{
    double n, m, w;
    TriangleInfo tri_info;

    n = cg->nVertices;
    tri_info = betterWedgeEnumerator(&(dag->outlist));

    for (VertexIdx i = 0; i < n; i++) // looping over vertices
    {
        VertexIdx deg = cg->offsets[i+1] - cg->offsets[i]; // degree of i
        orbit_counts.per_vertex_[0][i] = deg;
        //orbit_counts.per_vertex_[1][i] = 0;
        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; pos++) // loop over neighbors of i
        {
        	VertexIdx j = cg->nbors[pos]; // node j is the neighbor of node i
        	VertexIdx deg_j = cg->offsets[j+1] - cg->offsets[j]; // degree of j
        	orbit_counts.per_vertex_[1][i] += deg_j - 1;
        }
        orbit_counts.per_vertex_[2][i] = choosec(deg, 2);
        //OLD OLD OLD 
        // orbit_counts.per_vertex_[2][i] = (deg * (deg-1)) / 2;
        //OLD OLD OLD
        orbit_counts.per_vertex_[3][i] = tri_info.perVertex[i];
    }
    three_edge_orbit_counter(&(dag->outlist), orbit_counts);
}

// This function generates all node orbit counts for orbits in 4-vertex patterns.
// It is a wrapper function that calls the main algorithmic parts

// Input: pointer to CGraph, corresponding DAG, OrbitInfo structure to store orbit counts
// No output: orbit_counts will have all the orbit and edge orbit counts

void  get_all_four_orbits_node_only(CGraph *cg, CDAG *dag, OrbitInfo &orbit_counts)
{
    //cout << "node only here" << endl;
    double n, m, w;

    n = cg->nVertices;

    /*
    * OLD OLD OLD
    *
   for(EdgeIdx i = 0; i < (&(dag->outlist))->nEdges; i++)
   {
        for(int j=2; j < 12; j++)
        {
            orbit_counts.per_edge_[j][i] = 0;
            orbit_counts.per_edge_rev_[j][i] = 0;
        }
   }
   *
   * OLD OLD OLD
   */

    for (VertexIdx i = 0; i < n; i++) // loop over vertices
    {

        // OLD OLD OLD        
        // for(int j=4; j < 15; j++)
        // {
        //  orbit_counts.per_vertex_[j][i] = 0; 
        // }
        // OLD OLD OLD

        orbit_counts.per_vertex_[5][i] = orbit_counts.per_vertex_[1][i] * (orbit_counts.per_vertex_[0][i] - 1) - 2 * orbit_counts.per_vertex_[3][i];
        orbit_counts.per_vertex_[7][i] = choosec(orbit_counts.per_vertex_[0][i], 3);
        orbit_counts.per_vertex_[11][i] = orbit_counts.per_vertex_[3][i] * (orbit_counts.per_vertex_[0][i] - 2);

        
        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; pos++) // loop over neighbors of i
        {
            VertexIdx j = cg->nbors[pos];
            VertexIdx deg_j = orbit_counts.per_vertex_[0][j]; //degree of j
            orbit_counts.per_vertex_[4][i] += orbit_counts.per_vertex_[1][j];
            orbit_counts.per_vertex_[6][i] += choosec(deg_j - 1, 2);
        }

        orbit_counts.per_vertex_[4][i] -= 2 * orbit_counts.per_vertex_[2][i] + 2 * orbit_counts.per_vertex_[3][i];
    }


	cout << std::setprecision(6) << std::fixed;
    
    // otherFourPatternOrbitCounter
    clock_t start = clock();

    cout << "\ncounting other four orbits" << endl;
    otherFourPatternOrbitCounterNodeOnly(&(dag->outlist), &(dag->inlist), orbit_counts);

    clock_t end = clock();
    double duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    double max_duartion = duration;
    string longest_step = "otherFourPatternOrbitCounter";
    cout << "Time for otherFourPatternOrbitCounter: " << duration << endl;


    // four_cycle_orbit_counter
    start = clock();

    cout << "counting Four cycles" << endl;
    four_cycle_orbit_counter_node_only(&(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "four_cycle_orbit_counter";
    }
    cout << "Time for four_cycle_orbit_counter: " << duration << endl;

    
    // fourCliqueOrbitCounter
    start = clock();

    cout << "counting Four cliques" << endl;
    fourCliqueOrbitCounterNodeOnly(&(dag->outlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "fourCliqueOrbitCounter";
    }
    cout << "Time for fourCliqueOrbitCounter: " << duration << endl;

    cout << "\nTime for the most expensive orbit in 4-vertex patterns: " << max_duartion << endl;
    cout << longest_step << " took the longest time amongs orbits of 4-vertex patterns.\n" << endl;
}

// This function generates all orbit counts and edge orbit counts for orbits and edge orbits in 4-vertex patterns.
// It is a wrapper function that calls the main algorithmic parts

// Input: pointer to CGraph, corresponding DAG, OrbitInfo structure to store orbit counts
// No output: orbit_counts will have all the orbit and edge orbit counts

void  get_all_four_orbits(CGraph *cg, CDAG *dag, OrbitInfo &orbit_counts)
{
	double n, m, w;

    n = cg->nVertices;

    /*
    * OLD OLD OLD
    *
   for(EdgeIdx i = 0; i < (&(dag->outlist))->nEdges; i++)
   {
   	    for(int j=2; j < 12; j++)
    	{
    		orbit_counts.per_edge_[j][i] = 0;
    		orbit_counts.per_edge_rev_[j][i] = 0;
    	}
   }
   *
   * OLD OLD OLD
   */

    for (VertexIdx i = 0; i < n; i++) // loop over vertices
    {

        // OLD OLD OLD        
    	// for(int j=4; j < 15; j++)
    	// {
    	// 	orbit_counts.per_vertex_[j][i] = 0;	
    	// }
        // OLD OLD OLD

      	orbit_counts.per_vertex_[5][i] = orbit_counts.per_vertex_[1][i] * (orbit_counts.per_vertex_[0][i] - 1) - 2 * orbit_counts.per_vertex_[3][i];
      	orbit_counts.per_vertex_[7][i] = choosec(orbit_counts.per_vertex_[0][i], 3);
      	orbit_counts.per_vertex_[11][i] = orbit_counts.per_vertex_[3][i] * (orbit_counts.per_vertex_[0][i] - 2);

        
        for (EdgeIdx pos = cg->offsets[i]; pos < cg->offsets[i+1]; pos++) // loop over neighbors of i
        {
            VertexIdx j = cg->nbors[pos];
        	VertexIdx deg_j = orbit_counts.per_vertex_[0][j]; //degree of j
        	orbit_counts.per_vertex_[4][i] += orbit_counts.per_vertex_[1][j];
        	orbit_counts.per_vertex_[6][i] += choosec(deg_j - 1, 2);
        }

        orbit_counts.per_vertex_[4][i] -= 2 * orbit_counts.per_vertex_[2][i] + 2 * orbit_counts.per_vertex_[3][i];
    }


    cout << std::setprecision(6) << std::fixed;
    
    // otherFourPatternOrbitCounter
    clock_t start = clock();

    cout << "\ncounting other four orbits" << endl;
    otherFourPatternOrbitCounter(&(dag->outlist), &(dag->inlist), orbit_counts);

    clock_t end = clock();
    double duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    double max_duartion = duration;
    string longest_step = "otherFourPatternOrbitCounter";
    cout << "Time for otherFourPatternOrbitCounter: " << duration << endl;


    // four_cycle_orbit_counter
    start = clock();

    cout << "counting Four cycles" << endl;
    four_cycle_orbit_counter(&(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "four_cycle_orbit_counter";
    }
    cout << "Time for four_cycle_orbit_counter: " << duration << endl;

    
    // fourCliqueOrbitCounter
    start = clock();

    cout << "counting Four cliques" << endl;
    fourCliqueOrbitCounter(&(dag->outlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "fourCliqueOrbitCounter";
    }
    cout << "Time for fourCliqueOrbitCounter: " << duration << endl;

    cout << "\nTime for the most expensive orbit in 4-vertex patterns: " << max_duartion << endl;
    cout << longest_step << " took the longest time amongs orbits of 4-vertex patterns.\n" << endl;
}



void get_all_five_orbits(CGraph *cg, CDAG *dag, OrbitInfo &orbit_counts)
{

 	double n, m, w;

    n = cg->nVertices;

    TriangleInfo tri_info = betterWedgeEnumerator(&(dag->outlist));
    TriangleList allTris = storeAllTriangles(cg,tri_info.total);
    
    // OLD OLD OLD
	// for (VertexIdx i = 0; i < n; i++)
 //    {
 //    	for(int j=15; j < 73; j++)
 //    	{
 //    		orbit_counts.per_vertex_[j][i] = 0;	
 //    	}
 //    }
    // OLD OLD OLD

    cout << std::setprecision(6) << std::fixed;
    // FiveTreeOrbitCounter
    clock_t start = clock();
    
    cout << "\ncounting five trees" << endl;
    FiveTreeOrbitCounter(&(dag->outlist), &(dag->inlist), orbit_counts);

    clock_t end = clock();
    double duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    double max_duartion = duration;
    string longest_step = "FiveTreeOrbitCounter";
    // cout << "Time for FiveTreeOrbitCounter: " << ((double) (end - start)) / CLOCKS_PER_SEC << endl;
    cout << "Time for FiveTreeOrbitCounter: " << duration << endl;
    
    
    // FiveCycleOrbitCounter
    start = clock();

    cout << "counting five cycles" << endl;
    FiveCycleOrbitCounter(&(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "FiveCycleOrbitCounter";
    }
    cout << "Time for FiveCycleOrbitCounter: " << duration << endl;


    // WedgeCollisionOrbitCounter
    start = clock();

    cout << "counting wedge collision patterns" << endl;
    WedgeCollisionOrbitCounter(cg, &(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "WedgeCollisionOrbitCounter";
    }
    cout << "Time for WedgeCollisionOrbitCounter: " << duration << endl;
    

    // Pattern51OrbitCounter
    start = clock();

    cout << "counting orbit 51" << endl;
    Pattern51OrbitCounter(cg, &(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "Pattern51OrbitCounter";
    }
    cout << "Time for Pattern51OrbitCounter: " << duration << endl;

    
    // OtherFiveOrbitsCounter
    start = clock();

    cout << "counting other orbits" << endl;
    OtherFiveOrbitsCounter(&(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "OtherFiveOrbitsCounter";
    }
    cout << "Time for OtherFiveOrbitsCounter: " << duration << endl;


    // Pattern36OrbitCounter
    start = clock();

    cout << "counting orbit 36" << endl;
    Pattern36OrbitCounter(&(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "Pattern36OrbitCounter";
    }
    cout << "Time for Pattern36OrbitCounter: " << duration << endl;

    // ChordalWedgeColOrbitCounter
    start = clock();

    cout << "counting chordal wedge orbits" << endl;
    ChordalWedgeColOrbitCounter(cg, &(dag->outlist), &(dag->inlist), &allTris, orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "ChordalWedgeColOrbitCounter";
    }
    cout << "Time for ChordalWedgeColOrbitCounter: " << duration << endl;


    // Pattern66OrbitCounter
    start = clock();

    cout << "counting orbit 66" << endl;
    Pattern66OrbitCounter(&(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "Pattern66OrbitCounter";
    }
    cout << "Time for Pattern66OrbitCounter: " << duration << endl;

    // AlmostFiveCliqueOrbitCounter
    start = clock();

    cout << "counting almost five cliques" << endl;
    AlmostFiveCliqueOrbitCounter(cg, &(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "AlmostFiveCliqueOrbitCounter";
    }
    cout << "Time for AlmostFiveCliqueOrbitCounter: " << duration << endl;

    // FiveCliqueOrbitCounter
    start = clock();

    cout << "counting five cliques" << endl;
    FiveCliqueOrbitCounter(&(dag->outlist), &(dag->inlist), orbit_counts);

    end = clock();
    duration = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(duration > max_duartion)
    {
        max_duartion = duration;
        longest_step = "FiveCliqueOrbitCounter";
    }
    cout << "Time for FiveCliqueOrbitCounter: " << duration << endl;

    cout << "\nTime for the most expensive orbit in 5-vertex patterns: " << max_duartion << endl;
    cout << longest_step << " took the longest time amongs orbits of 5-vertex patterns.\n" << endl;
}

#endif
