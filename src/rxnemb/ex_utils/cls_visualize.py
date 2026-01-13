"""
qq:
This file contains utility functions for visualizing clustering results.
"""

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

__all__ = ["pairwise_class_distance", "reorder_by_optimal_leaf_ordering"]


def pairwise_class_distance(embd, labels, metric="euclidean"):
    """
    Parameters:
    - embd: (N, D) embedding matrix
    - labels: (N,) integer label vector
    - metric: distance metric, default 'euclidean', options include 'cosine', 'manhattan', etc.

    Returns:
    - dist_matrix: (K, K) distance matrix, where K is the number of classes
    """
    # Get class information
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    # Initialize distance matrix
    dist_matrix = np.zeros((K, K))

    # Group samples by class
    class_samples = {}
    for label in unique_labels:
        class_samples[label] = embd[labels == label]

    # Calculate average distance between each pair of classes
    for i, label_i in enumerate(tqdm(unique_labels)):
        for j, label_j in enumerate(unique_labels):
            if j < i:
                continue
            # Calculate all pairwise distances between samples from two classes
            samples_i = class_samples[label_i]
            samples_j = class_samples[label_j]

            pairwise_dists = pairwise_distances(samples_i, samples_j, metric=metric)

            dist_matrix[i, j] = np.mean(pairwise_dists)
            dist_matrix[j, i] = np.mean(pairwise_dists)

    return dist_matrix


def reorder_by_optimal_leaf_ordering(pairwise_dist_matrix):
    # Hierarchical clustering
    # This algorithm requires diagonal elements to be 0
    pairwise_dist_matrix = pairwise_dist_matrix.copy()
    np.fill_diagonal(pairwise_dist_matrix, 0)
    linkage_matrix = linkage(squareform(pairwise_dist_matrix), method="average")

    # Optimal leaf ordering
    optimal_linkage = optimal_leaf_ordering(linkage_matrix, squareform(pairwise_dist_matrix))
    optimal_order = leaves_list(optimal_linkage)

    return optimal_order
