# Orbit-Counting

Instructions:

1) Run "make" in main directory.

2) Go to wrappers/ directory.

3) For subgraph counting, run command:
	
	python3 subgraph_counts.py <PATH TO INPUT> <DESIRED PATTERN SIZE> <OPTIONAL FLAGS>
	
DESIRED PATTERN SIZE: Either 3, 4, or 5
OPTIONAL FLAGS:  
	-i: output counts as integers. Useful for small graphs, or for debugging
(For example, python3 subgraph_counts.py ../graphs/ca-AstroPh.edges 4 -i)
	
Input file must be in escape format, where first line has number 
of nodes and edges, and every line has a distinct undirected edge.

The console output gives the various numbers of patterns.
Each line has a different count in the format: pattern name, number of non-induced counts, number of induced counts, ratio of induced to non-induced

4) For node orbit counting, run command:

	python3 orbit_counts.py <PATH TO INPUT> <DESIRED PATTERN SIZE> <OPTIONAL FLAGS>
	
DESIRED PATTERN SIZE: Either 3, 4, or 5
OPTIONAL FLAGS:  
	-i: output orbit counts as integers. Useful for small graphs, or for debugging  
	-p: print orbit counts to the screen  
	-c: save induced counts to the out.txt file instead of non-induced counts
(For example, python3 orbit_counts.py ../graphs/smaller-test.edges 5 -i -p)

Input file must be in escape format, where first line has number 
of nodes and edges, and every line has a distinct undirected edge.

The console output gives the node orbit counts.
Line i is a list containing the counts of the times each node appears as the i-th node orbit.

-----------------------------------
Here are instructions to convert a raw list of edges (with potential repeats) to the escape format.

1) Go to directory python/
2) Run "python sanitize.py <DIR NAME> <EDGE FILE NAME>
       (eg python sanitize.py ../graphs ca-AstroPh.txt)

This will create a .edges file in the directory. Such a file will be in escape format.

-----------------------------------

graphs/ :

The graphs/ directory has a few raw input files. There
is also a small test graph called small-test.txt. Remember to sanitize these by calling "sanitize.py"
to convert into the escape format.

-----------------------------------

escape/ :

This directory has the main graph manipulation and pattern and orbit counting codes.

-----------------------------------

exe/ :

This directory has the main escape executables.

-----------------------------------

wrappers/ :

This directory has python wrapper functions through which the main code is called.

-----------------------------------

tests/ :

This directory has various test scripts, used for debugging and timing.


