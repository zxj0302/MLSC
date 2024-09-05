import os
import sys

folder = sys.argv[1]

filenames = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ("edges") ):
        filenames.append(filename)

filenames.sort()

with open("test_out.txt", 'w') as test_out_file:
	test_out_file.write("testing all the graphs with " + sys.argv[1].split('/')[2].split('_')[0] + " nodes")

for file_name in filenames:
	if file_name.endswith(".edges"):
		print(file_name)
		os.system("python3 test.py " + folder + file_name + " 5 -i -p")