# read the json file 
# read the graph file
# extract size of each graph in the test set
# calculate the gt count for each graph
# calculate the predicted count for each graph
# calculate the MAE
# calculate the Q-ERROR
# save the results
import os
import json
import torch

# RElu function
def RELU(x):
    return max(0, x)

def main():
    DATASET = "Set_1"
    res = json.load(open("/home/zxj/Dev/MLSC/output/few_shot_star_graph.json", "r"))
    for algo in ["ESC-GNN"]:
        res[algo] = {}
        res[algo][DATASET] = {}
        for target in [0,3,10]:
            res[algo][DATASET][target] = {}
            # /home/zxj/Dev/MLSC/output/retrain/Set_1/ESC-GNN/target_8.json
            json_file = f"/home/zxj/Dev/MLSC/code/ESC-GNN/output/fine/Set_1/ESC-GNN/{target}/300_cpt_test.json"
            graph_file = f"input/{DATASET}/dataset.pt"
            
            print(f"Processing {json_file}")

            with open(json_file, 'r') as f:
                data = json.load(f)
                pred = data["predictions"]
                gt = data["ground_truth"]
            # load the graph file
            graph_data = torch.load(graph_file, weights_only=False)
            test_data = graph_data['test']
            slices = [i['num_nodes'] for i in test_data]
            # calculate the gt count for each graph
            target_size = 5
            pred_graphs = []
            gt_graphs = []
            start = 0
            for i in slices:
                pred_graphs.append(RELU(sum(pred[start:start+i]))/target_size)
                gt_graphs.append(RELU(sum(gt[start:start+i]))/target_size)

                start += i
            res[algo][DATASET][target]["pred"] = pred_graphs
            res[algo][DATASET][target]["gt"] = gt_graphs

            print(pred_graphs)
            print("----------------------------------")

            # save the results
            with open(f"output/few_shot_star_graph.json", "w") as f:
                json.dump(res, f, indent=2)
            


if __name__ == "__main__":
    main()