# Data directory explanation 
1. all_noniso_graphs: All the non-isomorphic graphs with nodes in range [2, 9] in a specific format. To convert the to nx objects, use the `noniso_graphs_to_nx` function.
2. patterns: all pattern files are in the `data/patterns ` directory.
    * triangle: named `p_htw3_3_3_17` in the directory.
    * star: named `p_htw3_3_3_16` in the directory.
    * rectangle with an edge: named `p_htw3_4_5_3` in the directory.
    * pentagon: named `p_htw3_5_5_19` in the directory.
    * pentagon with an edge: named `p_htw3_5_5_20` in the directory.
3. datasets: all data graphs are in the corresponding `data/DATASET_NAME` directory, including
    * `g8_graphs`
    * `ogbg_molhiv`
    * `zinc`
    * `qm9` 
4. forbidden_minors: The 5 forbidden minor graphs of graphs with [treewidth=3](https://en.wikipedia.org/wiki/Treewidth#Forbidden_minors) are in the `data/forbidden_minors` directory.
    
Note: 
The g8_graphs (called `N8Graphs` in the paper) has been concluded in the `data/all_noniso_graphs` directory.
Other datasets can be downloaded from public urls using the following command:
```python
python -m gmatch.preprocessing -o generate --dataset ogbg_molhiv # download the ogbg_molhiv dataset to the data/ogbg_molhiv/ directory.
```

# Install
To run the code, editable installation is required.
```bash
cd /repo_root_directory
pip install -e . # install the code as a module in an editable mode.
chmod +x ./src/gmatch/subcounting/SubgraphMatching/build/matching/SubgraphMatching.out
```
# Preprocess
## Prepare graphs (download datasets)
`Prepare graphs` aims to download other three public graph datasets.
The download relies on the PyG and OGB library.
```python
python -m gmatch.preprocessing -o generate --dataset DATASET_NAME
```
This will downloads all graph files of the `DATASET_NAME` dataset in .graph format in `data/DATASET_NAME` directory.


## Process graphs
`Process graphs` aims to generate subcount labels for LPP and other baselines to train and test their performance.
This step will require the designation of both a dataset and a pattern.
```bash
python -m gmatch.preprocessing -o process --dataset DATASET_NAME --pattern_name PATTERN_NAME
```
The processing mainly consists of two phases:

1. Count number of the pattern in each graph as train/test labels.
2. Split graphs into train/val/test sets, and build a index file used in the WholeGraphSampler sampler (used by LPP model).


## Train & test
Train and test the model with a dataset and a pattern, e.g.,
```bash
python main.py --dataset g8_graphs --pattern p_htw3_3_3_16 --model cnn --epoch 1 --batch_size 4 --device cuda:0
```

## Convert datasets formats
`Convert datasets formats` aims to convert the graphs in `data/DATASET_NAME` directory to the data formats required by other baselines, including `GIN/GCN`, `PPGN`, and `LRP`.
For example, if we want to validate the performance of `LRP` for substructure counting,
we need to:
1. Clone the code of LRP in the directory which the LPP lies in.
2. Convert a dataset to LRP's input format.
3. Run the LRP code.

```bash
python -m gmatch.converter -m model_name -d dataset -p pattern # model_name can be GIN, PPGN, LRP.
```

## Show statistics of counts for a dataset and a pattern
The statistics figures will be saved in the `figures/` directory.
```bash
python -m gmatch.statistics -d dataset -p pattern
```