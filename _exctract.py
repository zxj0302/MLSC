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

def extract_graph_slices(dataset="Set_1"):
    graph_file = f"input/{dataset}/dataset.pt"
    graph_data = torch.load(graph_file, weights_only=False)
    test_data = graph_data['test']
    slices = [i['num_nodes'] for i in test_data]
    return slices

def load_pred_gt(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        pred = data["predictions"]
        gt = data["ground_truth"]
    return pred, gt

def calculate_graph_count(pred, gt, slices, target_size=5):
    pred_graphs = []
    gt_graphs = []
    start = 0
    for i in slices:
        pred_graphs.append(RELU(sum(pred[start:start+i]))/target_size)
        gt_graphs.append(RELU(sum(gt[start:start+i]))/target_size)

        start += i
    return pred_graphs, gt_graphs

def avg_mae(pred, gt):
    return sum([abs(p-g) for p, g in zip(pred, gt)])/len(pred)

def q_error(pred, gt):
    return sum([max((p+1)/(g+1), (g+1)/(p+1)) for p, g in zip(pred, gt)])/len(pred)

def target_to_size(target):
    if target < 2:
        return 3
    elif target < 8:
        return 4
    return 5

def esc_trends():
    target = 11
    DATASET = "Set_1"
    epochs = 1000
    res = json.load(open("/home/zxj/Dev/MLSC/output/graph_error_per_epoch.json", "r"))
    res["ESC-GNN"] = {
        "11":{"mae":[], "q_error":[]}
    }
    slices = extract_graph_slices(DATASET)
    for ep in range(1, epochs+1):
        json_file = f"/home/zxj/Dev/MLSC/code/ESC-GNN/output/fine/Set_1/ESC-GNN/11/{ep}_cpt_test.json"
        print(f"Processing {json_file}")
        pred, gt = load_pred_gt(json_file)
        target_size = target_to_size(target)
        pred_graphs, gt_graphs = calculate_graph_count(pred, gt, slices, target_size)
        mae = avg_mae(pred_graphs, gt_graphs)
        q = q_error(pred_graphs, gt_graphs)
        res["ESC-GNN"]["11"]["mae"].append(mae)
        res["ESC-GNN"]["11"]["q_error"].append(q)
        print(f"MAE: {mae}, Q-ERROR: {q}")
        # with open(f"output/graph_error_per_epoch.json", "w") as f:
        #     json.dump(res, f, indent=2)


def desco_trends():
    mae_file = "/home/zxj/Dev/MLSC/output/desco_fine_trend_s1_mae.json"
    q_file = "/home/zxj/Dev/MLSC/output/desco_fine_trend_s1_qerrors.json"
    res = json.load(open("/home/zxj/Dev/MLSC/output/graph_error_per_epoch.json", "r"))
    
    mae_load = json.load(open(mae_file, "r"))
    q_load = json.load(open(q_file, "r"))
    res["DESCO"] = {
        "11":{"mae":[
            mae_load[11]
        ], 
        "q_error":[
            q_load[11]
        ]}
    }
    # save the results
    with open(f"output/graph_error_per_epoch.json", "w") as f:
        json.dump(res, f, indent=2)


def error_graph_target11_retrain():
    json_file = "/home/zxj/Dev/MLSC/code/ESC-GNN/output/retrain/Set_1/ESC-GNN/11/2000_cpt_test.json"
    slices = extract_graph_slices("Set_1")
    pred, gt = load_pred_gt(json_file)
    target_size = target_to_size(11)
    pred_graphs, gt_graphs = calculate_graph_count(pred, gt, slices, target_size)
    res = json.load(open("/home/zxj/Dev/MLSC/output/graph_error_target11_full_retrain.json", "r"))
    res["ESC-GNN"] = {
        "Set_1": {"pred": pred_graphs, "gt": gt_graphs}
    }
    # save the results
    with open(f"output/graph_error_target11_full_retrain.json", "w") as f:
        json.dump(res, f, indent=2)

def error_node_target11_retrain():
    json_file = "/home/zxj/Dev/MLSC/code/ESC-GNN/output/retrain/Set_1/ESC-GNN/11/2000_cpt_test.json"
    pred, gt = load_pred_gt(json_file)
    res = json.load(open("/home/zxj/Dev/MLSC/output/node_error_target11_full_retrain.json", "r"))
    relu_pred = [RELU(p) for p in pred]
    relu_gt = [RELU(g) for g in gt]
    res["ESC-GNN"] = {
        "Set_1": {"pred": relu_pred, "gt": relu_gt}
    }
    # save the results
    with open(f"output/node_error_target11_full_retrain.json", "w") as f:
        json.dump(res, f, indent=2)

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
    error_graph_target11_retrain()
    error_node_target11_retrain()