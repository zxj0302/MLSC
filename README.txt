The structure of the project is as follows:

MLSC:
    input(sampled and split in run.py, Set_i is different dataset):
        rwd.db(oracle dataset)
        Set_1:
            config.json(for sampler)
            dataset.pt(a dict or list, contains train, val and test)
        Set_2:
        Set_3:
    output:
        Set_1:
            DeSCo:
                pretrained:
                    log.txt(e.g. time, command used)
                    prediction_graph.csv
                    prediction_node.csv
                fine-tuning:
                fine-tuned:
            ESC-GNN:
                params_1:
                    log.txt(must have time for each part and hyper-parameters)
                    prediction_graph.csv
                    prediction_node.csv
                params_2:
            EVOKE:
            MOTIVO:
        Set_2:
    plots:
    code:
        DeSCo(contains a script that read in sampled dataset in MLSC/input and output the needed files to MLSC/output):
        ESC-GNN:
        ESCAPE:
    config.json(config for run.py)
    run.py(
        1. sample for Set_i. 
        2. analyze the input(sampled datasets) and plot. 
        3. create containers and run all models on all Set_i.
        4. analyze the output(predictions & truth on node & graph) and plot.
        )
