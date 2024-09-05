import os
import numpy as np
import numpy.matlib
import sys
import math

# This code gives each of the patterns with 5 vertices as input to the orbit counting code
# and gets the results for each orbit per each node of the pattern
# which gives us the matrix to convert the induced counts to non-induced counts

five_conversion_matrix = np.zeros((58, 58))
filled = [False] * 73

orbits_in_patterns = np.zeros((30,5))	


orbits_in_patterns[9]  = [15, 16, 17, 16, 15]
orbits_in_patterns[10] = [18, 20, 21, 19, 19]
orbits_in_patterns[11] = [23, 22, 22, 22, 22]
orbits_in_patterns[12] = [25, 26, 26, 24, 24]
orbits_in_patterns[13] = [27, 28, 30, 29, 29]
orbits_in_patterns[14] = [32, 32, 33, 31, 31]
orbits_in_patterns[15] = [34, 34, 34, 34, 34]
orbits_in_patterns[16] = [36, 37, 37, 38, 35]
orbits_in_patterns[17] = [41, 40, 40, 42, 39]
orbits_in_patterns[18] = [43, 43, 44, 43, 43]
orbits_in_patterns[19] = [46, 48, 48, 47, 45]
orbits_in_patterns[20] = [49, 50, 49, 50, 49]
orbits_in_patterns[21] = [52, 53, 53, 51, 51]
orbits_in_patterns[22] = [54, 55, 55, 54, 54]
orbits_in_patterns[23] = [57, 57, 57, 58, 56]
orbits_in_patterns[24] = [59, 61, 60, 60, 59]
orbits_in_patterns[25] = [64, 63, 64, 63, 62]
orbits_in_patterns[26] = [66, 66, 67, 67, 65]
orbits_in_patterns[27] = [68, 68, 69, 68, 68]
orbits_in_patterns[28] = [70, 71, 71, 71, 70]
orbits_in_patterns[29] = [72, 72, 72, 72, 72]

#np.set_printoptions(threshold=np.inf)


for inp_file in sorted(os.listdir('./inputs/'), key=lambda x: (len(x),x)):
	print(str(int(inp_file.split('.')[0][1:])))
	os.system('../exe/count_orbit_five '+ "./inputs/"+inp_file)
	with open('./out.txt', 'r') as res_file:
		node_orbits = np.array([[int(x) for x in line.split()] for line in res_file])
		pattern = int(inp_file.split('.')[0][1:])
		for orbit_index in range(5):
			if(filled[int(orbits_in_patterns[pattern][orbit_index])] == False):
				filled[int(orbits_in_patterns[pattern][orbit_index])] = True
				five_conversion_matrix[:,int(orbits_in_patterns[pattern][orbit_index])-15] = node_orbits[15:,orbit_index]

#print(five_conversion_matrix)

with open("./five_conversion_matrix.txt", 'w') as res_file:
	for row in range(58):
		res_file.write("\'")
		for col in range(58):
			res_file.write(str(five_conversion_matrix[row,col])+" ")
		res_file.write(";\'\n")

