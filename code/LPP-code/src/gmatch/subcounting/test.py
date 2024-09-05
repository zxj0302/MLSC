import sys
import os
from utils import subgraph_count


if __name__ == "__main__":
    # input_binary_path = sys.argv[1]
    input_binary_path = './SubgraphMatching/build/matching/SubgraphMatching.out'
    if not os.path.isfile(input_binary_path):
        print('The binary {0} does not exist.'.format(input_binary_path) )
        exit(-1)
    graph_file = './SubgraphMatching/test/data_graph/HPRD.graph'
    pattern_file = './SubgraphMatching/test/query_graph/query_dense_16_1.graph'

    # input_binary_path = './SubgraphMatching/build/matching/SubgraphMatching.out'
    num = subgraph_count(input_binary_path, graph_file, pattern_file)
    print(f'Count num: {num}')

