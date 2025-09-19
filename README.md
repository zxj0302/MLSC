# MLSC Project Overview
## 0. Website
You can find more detailed information(including dataset, leaderboard, more experiment results, etc.) on our [website](https://momatinaj.github.io/beacon/).

## 1. BEACON-Oracle Dataset
A key motivation for our Oracle Dataset is the absence of a large-scale dataset with ground truth subgraph counts in the community. To address this gap, we collected all graphs from the TUDataset alongside graphs from the OGB dataset, which encompasses a variety of domains such as bioinformatics, social networks, and computer vision. For each graph, we computed the ground truth counts for all subgraphs with up to five nodes, considering both local and global frequencies as well as induced and non-induced configurations.

You can download our Oracle Dataset [here](https://drive.google.com/file/d/1lf2B3XBqAOrSVnLu7vXhezGbVRVP11nc/view?usp=sharing). We also provide the code for sampling the graphs from TUDataset and OGB and the code for ground truth computation so that you can build your own dataset, or extend our dataset to absorb more graphs or generalize to larger k. Our Oracle Dataset contains a total of 26,435 graphs drawn from multiple domains. In addition to the standard ground truth subgraph counts, we have augmented the dataset with the ID-constrained ground truth introduced by the DeSCo algorithm. To further characterize each graph, we computed several graph-level features including diameter, density, and clustering coefficient.

## 2. BEACON-Sampler
Given that the Oracle Dataset comprises around 24,000 graphs with a wide range of characteristics, it is often impractical to utilize the full dataset in every research application. In many cases, researchers require a carefully selected subset that meets specific experimental criteria. For example, a study might need a sample of 10 social networks with a density greater than 3 and a node count between 20,000 and 50,000. To meet these tailored needs, we have developed the BEACON-Sampler, a tool that enables users to efficiently downsample the Oracle Dataset based on their given constraints.

The BEACON-Sampler is a versatile tool for extracting graphs from a database based on specific structural and numerical criteria. Publicly available on [PyPI](https://pypi.org/project/rwdq/) under the name rwdq, this tool enables researchers to tailor their dataset selection through a JSON configuration file. Users can specify constraints such as minimum and maximum node counts, average degree thresholds, and other key graph properties.

### Usage:

```python
python main.py --config my_config.json --database my_database.db
data_objects = run_query('config.json', 'rwd.db')
```

## 3. Docker Images
### Description
The dockerhub [rep](https://hub.docker.com/repository/docker/zhuxiangju/benchmark_subgraphcounting/general) contains many envs for ML/GNN based subgraph counting methods, e.g. DeSCo and ESC-GNN. Also, there is a basic image, which can be used for ESCAPE, EVOKE and maybe other classical subgraph counting methods. The MOTIVO image built on Ubuntu-20.04(got an error on Ubuntu-22.04, seems Ubuntu-22.04 uses newer version of some library). The SCOPE image built on the MOTIVO image with GLPK library installed.

### Images Information
* Basic image: cuda 12.5, python 3.12.3 in conda base env, torch 2.3.1, torch_geometric 2.5.3, lightning 2.3.0, gcc/g++ 11.4, openssh-server installed(no auto-start on login), can be used to develop new methods/models.
* DeSCo image: all packages align with the requirements.txt in their github rep. This is a must, especially the python version 3.9 and pytorch_lightning version 1.6.4.
* Motivo image: ubuntu 20.04, gcc/g++ 9.4.0, bc(caulculator needed by scripts/motivo.sh) installed.
* Scope image: GLPK library installed, built on Motivo image.
* ESC-GNN: all align with the requirements.txt(ubuntu 20.04, cuda 11.1, python 3.7 in conda ESC-GNN env, torch 1.8.0, torch_geometric 2.0.4) but pycairo. Can also support MPNN, PPGN, I2GNN, IDGNN, GNNAK+ and other models in their code.

### Usage
You can create a container like this(I put --cpus=1 here to avoid auto-parallel when loading datasets):
```
> docker run -it -v (YOUR_FOLDER_PATH, i.e. /home/usr_name/codelib):(MOUNT_PATH, i.e. /workspace) -p (HOST_PORT, i.e. 9999):(CONTAINER_PORT, i.e. 22) --name (CONTAINER_NAME, i.e. DeSCo) --gpus (GPU_SET, i.e. all) --cpus=(NUM_CPU, i.e. 1) --ipc=host (IMAGE_NAME, i.e. zhuxiangju/benchmark_subgraphcounting:DeSCo_LINUX)
```
After starting the container, you can change the sshd_config file to allow root login, config passwd root, and then you can simply 'ssh root@127.0.0.1 -p 9999' to connect to the container.

**Note**: We created the images with docker desktop + WSL2. It is expected to run well on WSL2, but for transferring to linux, there are some problems. If you are using --gpus all for DeSCo, ESC-GNN or other GPU-related images, GPU driver(s) on host cannot be correctly linked. For more information, please read [CSDN](https://blog.csdn.net/qq_40243750/article/details/130661104), [Github](https://github.com/NVIDIA/nvidia-container-toolkit/issues/289), [Github](https://github.com/microsoft/WSL/issues/8274), and [Nvidia](https://github.com/microsoft/WSL/issues/8274). We will create a dockerfile for one-click use, for now you can run the /opt/softlink_remake.sh after creating the container as a temporary solution. Another thing needs notice is that, when running with 13th/14th or some 12th generation Intel CPU with P/E-cores, if you want to only use 1 CPU core or avoid multi-core to do benchmark, please make sure different containers or algorithms run on the same kind of CPU because the performance of P-cores are normally better than E-cores. You can assign CPU to container with '[--cpuset-cpus](https://docs.docker.com/config/containers/resource_constraints/#cpu)' option.

Update(14/08/2024): New images created with suffix('LINUX'|'WSL') for linux and wsl2 users respectively. No need to run the .sh file now.

Update(14/08/2024): '--ipc=host' added to solve the '[RuntimeError](https://github.com/ultralytics/yolov3/issues/283#issuecomment-552776535)' led by insufficient shared memory.

Update(06/09/2024): To run python scripts in the container, some packages(python, torch, torch_geometric, etc.) are added to image 'basic' and 'Motivo'. They are renamed to 'BASIC' and 'MOTIVO' respectively. Old images will be deleted.

## 4. Project Structure
As we want to keep everything simple and easy-to-use, we use json file to pass parameters, and the main workflow is:
1. Sample the dataset you want to use from the oracle dataset or use our pre-sampled Set_i for comparison.
2. Use our containers and mount the runfile and code repository to run experiments.
3. Plot or analyse the output results.


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
