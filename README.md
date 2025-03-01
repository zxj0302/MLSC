# MLSC Project Overview

## 1. BEACON-Oracle Dataset
A key motivation for our Oracle Dataset is the absence of a large-scale dataset with ground truth subgraph counts in the community. To address this gap, we collected all graphs from the TUDataset alongside graphs from the OGB dataset, which encompasses a variety of domains such as bioinformatics, social networks, and computer vision. For each graph, we computed the ground truth counts for all subgraphs with up to five nodes, considering both local and global frequencies as well as induced and non-induced configurations.

You can download our Oracle Dataset [here](https://drive.google.com/file/d/1lf2B3XBqAOrSVnLu7vXhezGbVRVP11nc/view?usp=sharing). We also provide the code for sampling the graphs from TUDataset and OGB and the code for ground truth computation so that you can build your own dataset, or extend our dataset to absorb more graphs or generalize to larger k. Our Oracle Dataset contains a total of 26,435 graphs drawn from multiple domains. In addition to the standard ground truth subgraph counts, we have augmented the dataset with the ID-constrained ground truth introduced by the DeSCo algorithm. To further characterize each graph, we computed several graph-level features including diameter, density, and clustering coefficient.

## 2. BEACON-Sampler
Given that the Oracle Dataset comprises around 24,000 graphs with a wide range of characteristics, it is often impractical to utilize the full dataset in every research application. In many cases, researchers require a carefully selected subset that meets specific experimental criteria. For example, a study might need a sample of 10 social networks with a density greater than 3 and a node count between 20,000 and 50,000. To meet these tailored needs, we have developed the BEACON-Sampler, a tool that enables users to efficiently downsample the Oracle Dataset based on their given constraints.

The BEACON-Sampler is a versatile tool for extracting graphs from a database based on specific structural and numerical criteria. Publicly available on [PyPI](https://pypi.org/project/rwdq/) under the name rwdq, this tool enables researchers to tailor their dataset selection through a JSON configuration file. Users can specify constraints such as minimum and maximum node counts, average degree thresholds, and other key graph properties.

## 3. Project Structure

### MLSC Directory Structure

**MLSC/**
- **input/** *(sample datasets are created and split in run.py, where each Set_i is a different dataset)*
  - `rwd.db` *(oracle dataset)*
  - **Set_1/**
    - `config.json` *(configuration file for sampling)*
    - `dataset.pt` *(dict or list containing train, val, and test splits)*
  - **Set_2/**
  - **Set_3/**

- **output/**
  - **Set_1/**
    - **DeSCo/**
      - **pretrained/**
        - `log.txt` *(e.g., includes time, command used)*
        - `prediction_graph.csv`
        - `prediction_node.csv`
      - **fine-tuning/**
      - **fine-tuned/**
    - **ESC-GNN/**
      - **params_1/**
        - `log.txt` *(must have time for each part and hyper-parameters)*
        - `prediction_graph.csv`
        - `prediction_node.csv`
      - **params_2/**
    - **EVOKE/**
    - **MOTIVO/**
  - **Set_2/**

- **plots/**

- **code/**
  - **DeSCo/** *(contains a script that reads in sampled dataset from MLSC/input and outputs the needed files to MLSC/output)*
  - **ESC-GNN/**
  - **ESCAPE/**

- `config.json` *(configuration for run.py)*
- `run.py`

### Detailed Notes

1. **`run.py`**  
   - **Step 1**: Sample for each `Set_i`.  
   - **Step 2**: Analyze the input (sampled datasets), then create and save plots.  
   - **Step 3**: Create containers (environments) and run all models on all `Set_i` configurations.  
   - **Step 4**: Analyze the output (predictions & ground truth on nodes & graphs) and plot the results.

2. **Input Datasets**  
   - `rwd.db` is the so-called **oracle dataset**.  
   - Each `Set_i` folder contains:  
     - `config.json`: Parameters and configurations for sampling.  
     - `dataset.pt`: Contains the train, validation, and test splits.

3. **Models**  
   - **DeSCo**  
     - `pretrained/`, `fine-tuning/`, and `fine-tuned/` directories hold logs and predictions.  
   - **ESC-GNN**  
     - Multiple parameter sets (e.g., `params_1`, `params_2`, etc.). Each contains logs (with timestamps and hyper-parameter settings) and predictions.  
   - **EVOKE**  
   - **MOTIVO**

4. **Outputs**  
   - For each `Set_i`, the corresponding output directory will contain subfolders for each model.  
   - Logging and prediction files stored here.  

5. **Plots**  
   - This folder contains all generated plots from `run.py` (including intermediate analyses and final results).

6. **Code Folder**  
   - Each subfolder (e.g., `DeSCo`, `ESC-GNN`, `ESCAPE`) houses scripts that read from `MLSC/input` and generate outputs in `MLSC/output`.

---

**End of README**
