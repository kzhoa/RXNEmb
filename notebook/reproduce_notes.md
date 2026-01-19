# Overview

This `.md` file contains instruction to reproduce our results in the paper.

# Guidance 

## Step 1: Clone the project

```bash
git clone https://github.com/kzhoa/RXNEmb
```

## Step 2: Switch to the `rep` branch

The `rep` branch contains the legacy version of our RXNEmb model used in the original experiments. 

Since the `main` branch contains the latest version of the model with updated implementations and potential modifications, you'll need to switch to the older version first to ensure accurate reproduction of our published results.

```bash
git switch rep
```

## Step 3: Prepare Environment

After switching to the `rep` branch, set up the Conda environment using the provided `environment.yml`:
 ```bash
conda env create -f environment.yml 
```


>**Note:**
>If `conda env create` fails (e.g., due to unresolved dependencies), you can manually create the environment and install dependencies via `pip`:
> 
> ```bash
> conda create -n rxnemb python=3.11 -y
> conda activate rxnemb
> pip install -r requirement_torch221.txt
> ```
> Make sure you are in the **/project/root/directory** before running `pip install`.


Finally, register the `rxnemb` environment as a Jupyter kernel so it appears in notebooks:
```bash
conda install ipykernel

python -m ipykernel install \
--user \
--name rxnemb \
--display-name "Python (rxnemb)"
```


## Step4: Download cached files

To reproduce our experimental results efficiently, you need to download the pre-computed files. These files are organized into three main categories and should be placed in `{project_root}/download/`.

### Directory Overview

This project organizes its files into three main directories:

- **`dataset`**: Contains the original dataset files
- **`weights`**: Stores model weight files  
- **`cached_results`**: Contains intermediate artifacts generated during data processing


### File Checklist

| Sign       | name                             | md5                              | Size | Remark   |
| ---------- | -------------------------------- | -------------------------------- | ---- | -------- |
| required   | cached_results/datas_50k.pkl     | 8cbb71214061004741c754ee3df4e2f1 | 336M | Figure 3 |
| (optional) | cached_results/clusterer_c50.pkl | 55de20f1ba56026df7459e8073c55843 | 19G  | Figure 3 |
| (optional) | dataset/50k_with_rxn_type.zip    | e78715477da5dca374ecf21678e6a986 | 6.6M | Figure 3 |
|            |                                  |                                  |      |          |
|            |                                  |                                  |      |          |


## Step 5: Run notebook via `rxnemb` environment

Now that the `rxnemb` environment is created and registered as a Jupyter kernel, you can launch and run the notebooks using this environment.

```bash
conda activate rxnemb
jupyter notebook
```

Once opened a notebook, check the top-right corner​ of the notebook interface — it should show the current kernel.
If it does not​ say "Python (rxnemb)", click the kernel name and select `Python (rxnemb)` from the dropdown menu.


