# Overview

This `.md` file contains instruction to reproduce our results in the paper.

# Step 1: Clone the project

```bash
git clone https://github.com/kzhoa/RXNEmb
```

# Step 2: Switch to the `rep` branch

The `rep` branch contains the legacy version of our RXNEmb model used in the original experiments. 

Since the `main` branch contains the latest version of the model with updated implementations and potential modifications, you'll need to switch to the older version first to ensure accurate reproduction of our published results.

```bash
git switch rep
```

# Step 3: Prepare Environment

After switching into `rep` branch, prepare the environment via `requirement_torch221.txt`;

```bash
conda create -n rxn python==3.11
conda activate rxn
```

after activation 
```bash
# cd the project root
pip install -r requirement_torch221.txt
```


Register the new `rxn` environment into Jupyter.
```bash
conda install ipykernel

python -m ipykernel install \
--user \
--name rxn \
--display-name "Python (rxn)"
```


# Step4: Download cached files

To reproduce our experimental results efficiently, you need to download the pre-computed files. These files are organized into three main categories and should be placed in `{project_root}/download/`.

## Directory Overview

This project organizes its files into three main directories:

- **`dataset`**: Contains the original dataset files
- **`weights`**: Stores model weight files  
- **`cached_results`**: Contains intermediate artifacts generated during data processing


## File Checklist

| Sign       | name                             | md5                              | Size | Remark   |
| ---------- | -------------------------------- | -------------------------------- | ---- | -------- |
| required   | cached_results/datas_50k.pkl     | 8cbb71214061004741c754ee3df4e2f1 | 336M | Figure 3 |
| (optional) | cached_results/clusterer_c50.pkl | 55de20f1ba56026df7459e8073c55843 | 19G  | Figure 3 |
| (optional) | dataset/50k_with_rxn_type.zip    | e78715477da5dca374ecf21678e6a986 | 6.6M | Figure 3 |
|            |                                  |                                  |      |          |
|            |                                  |                                  |      |          |


# Step 5: Run notebook via `rxn` environment

