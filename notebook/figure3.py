import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.rxnemb import RXNEMB
from src.rxnemb.ex_utils.cls_visualize import pairwise_class_distance, reorder_by_optimal_leaf_ordering
from src.rxnemb.ex_utils.ks_cluster import FarthestPointClustering, compute_pairwise_distances_batch


def merge_columns(df1, df2, col1, col2):
    if len(df1) != len(df2):
        raise ValueError(f"lens should match, got {len(df1)} and {len(df2)}")
    if isinstance(col1, list) and isinstance(col2, list):
        merged_df = pd.concat([df1[col] for col in col1] + [df2[col] for col in col2], axis=1)
    elif isinstance(col1, str) and isinstance(col2, str):
        merged_df = pd.DataFrame({col1: df1[col1], col2: df2[col2]})
    else:
        raise ValueError("col1 and col2 should either be both strings or both lists")
    return merged_df


def get_reaction_smiles():
    # data
    base_dir = Path("./download/50k_with_rxn_type/")
    df_rct = pd.read_csv(base_dir / "50k_rxn_type_rct_0.csv", header=None, names=["reactant", "label"])
    df_pdt = pd.read_csv(base_dir / "50k_rxn_type_pdt_0.csv", header=None, names=["product", "label"])
    df_50k = merge_columns(df_rct, df_pdt, ["reactant", "label"], ["product"])
    rxn_smiles = [f"{row['reactant']}>>{row['product']}" for i, row in df_50k.iterrows()]
    return df_50k, rxn_smiles


def gen_embd_from_50k(pretrain_model_path):
    # load data
    df_50k, rxn_smiles = get_reaction_smiles()

    # model
    model = RXNEMB(pretrained_model_path=pretrain_model_path, model_type="classifier")

    # generate embd
    embds = model.gen_rxn_emb(rxn_smiles)
    datas_50k = {"embeddings": embds, "labels": df_50k["label"].values, "rxn_smiles": rxn_smiles}
    return datas_50k, rxn_smiles


def main():
    # ==== Load reaction embeddings for 50k dataset ====

    # To get exatly the same embeddings as in the figure3, use the cached file
    datas_50k = pickle.load(open("download/cached_results/datas_50k.pkl", "rb"))
    rxn_emb = np.array(datas_50k["embeddings"])
    rxn_smiles = datas_50k["rxn_smiles"]
    print("loaded rxnemb", rxn_emb.shape)  # (50000, 768) expected
    print(len(rxn_smiles))  # 50,000 expected

    # If you prefer to generate the embeddings from scratch, uncomment below:
    # Note: Requires downloading pretrained models and switching to `rep` branch first.
    # refer to `notebook/reproduce_notes.md` for instructions.
    """
    pretrain_model_path = "./download/pretrained_classification_model"
    datas_50k, rxn_smiles = gen_embd_from_50k(pretrain_model_path)
    """

    print("Computing distance matrix...")
    dist_matrix = compute_pairwise_distances_batch(rxn_emb)
    print(f"Done. dist_matrix.shape = {dist_matrix.shape}")

    # Get threshold from full pairwise distance distribution
    threshold = 33.95
    print(f"Using threshold: {threshold}")
    clusterer_c50 = FarthestPointClustering(threshold)
    clusterer_c50.fit(rxn_emb, dist_mat=dist_matrix)

    # there should be 50 clusters under this threshold
    assert len(clusterer_c50.cluster_centers_) == 50

    # If this check failed, we also offerred an checkpoint to reproduce our results:
    # clusterer_c50 = pickle.load(open("download/cached_results/clusterer_c50.pkl","rb"))

    # You can adjust the threshold to get your desired number of clusters :D

    # ==== Make a summary of clustering results ====
    def print_cluster_results(clusterer):
        print(f"\n Cluster:")
        print(f"Num Clusters: {clusterer.n_clusters_}")
        print(f"Center indices: {clusterer.get_centers()}")

        unique_labels, counts = np.unique(clusterer.labels_, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples ({count/clusterer.dist_mat.shape[0]*100:.1f}%)")

    print_cluster_results(clusterer_c50)

    # === Table S1: Details of reactions within each group ===
    # Show the SMILES of cluster centers,
    # which correspond to the `Table S1` in Supplementary Information
    center_rec_smiles = [rxn_smiles[i] for i in clusterer_c50.cluster_centers_]
    print("center_rec_smiles:\n")
    for i, smi in enumerate(center_rec_smiles):
        print(f"Group {i+1}", smi)

    # ====  Visualize the clustering results ====
    # 1. Pairwise class distance matrix ====
    print("\nComputing class distance matrix...")
    cls_dist_matrix = pairwise_class_distance(rxn_emb, clusterer_c50.labels_)
    print("Done.")
    print("\nClass distance matrix Shape:", cls_dist_matrix.shape)  # (50, 50)

    new_order_leaf = reorder_by_optimal_leaf_ordering(cls_dist_matrix)
    new_cls_dist_matrix = cls_dist_matrix[new_order_leaf, :][:, new_order_leaf].copy()

    # This should reproduce the heatmap in Figure 3b
    plt.figure(figsize=(12, 10))
    sns.heatmap(new_cls_dist_matrix, cmap="RdYlBu_r", annot=False, vmin=24, vmax=42)
    # ==== End visualization ====
