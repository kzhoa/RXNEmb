"""
Implementation of Kennard-Stone algorithm optimized for large datasets.
This should be faster.

by qq
Jan. 2026
"""

from typing import Optional, Tuple

import numpy as np
from tqdm import trange


def pairwise_euclidean_vectorized(x, y):
    """Compute pairwise Euclidean distances using vectorized operations.

    Args:
        x: Input matrix of shape (m, d)
        y: Input matrix of shape (n, d)

    Returns:
        Distance matrix of shape (m, n)
    """
    sum_x = np.sum(x**2, axis=1, keepdims=True)  # (n, 1)
    sum_y = np.sum(y**2, axis=1)  # (n,)

    dist_sq = sum_x + sum_y - 2 * x @ y.T  # (n, n)
    dist_sq = np.maximum(dist_sq, 0)
    np.fill_diagonal(dist_sq, 0)
    return np.sqrt(dist_sq)


def compute_pairwise_distances_batch(embeddings, batch_size=1000):
    """Compute full pairwise distance matrix using memory-efficient batch processing.

    Args:
        embeddings: Feature matrix of shape (n, d)
        batch_size: Size of batches for memory management

    Returns:
        Pairwise distance matrix of shape (n, n)
    """
    n = len(embeddings)

    dist_matrix = np.zeros((n, n))

    for i in trange(0, n, batch_size):
        for j in range(0, n, batch_size):
            dist_batch = pairwise_euclidean_vectorized(embeddings[i : i + batch_size], embeddings[j : j + batch_size])
            dist_matrix[i : i + batch_size, j : j + batch_size] = dist_batch

    return dist_matrix


# def get_distance_threshold_full(pairwise_distances, percentile=75):
#     """Calculate distance threshold from complete pairwise distance distribution.

#     Args:
#         pairwise_distances: Full distance matrix
#         percentile: Percentile value for threshold calculation

#     Returns:
#         Distance threshold value
#     """
#     upper_triangle = pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)]
#     threshold = np.percentile(upper_triangle, percentile)
#     print(f"Statistics:")
#     print(f"Distance Range: [{np.min(upper_triangle):.4f}, {np.max(upper_triangle):.4f}]")
#     print(f"Average Dist: {np.mean(upper_triangle):.4f}")
#     print(f"Middle Dist: {np.median(upper_triangle):.4f}")
#     print(f"{percentile}% percentile: {threshold:.4f}")
#     return threshold


class FarthestPointClustering:

    def __init__(self, threshold: float):
        """
        Args:
            threshold: min distance between cluster centers
        """
        self.threshold = threshold
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_clusters_ = 0

    def _find_farthest_pair(self) -> Tuple[int, int, float]:
        """
        Returns:
            tuple: (index1, index2, max_distance)
        """
        max_distance = self.max_distance
        farthest_pair = np.unravel_index(np.argmax(self.dist_mat), self.dist_mat.shape)
        return farthest_pair[0], farthest_pair[1], max_distance

    def _find_next_center(
        self,
    ) -> Optional[int]:
        """
        Returns:
            int or None:
        """

        n_samples = len(self.dist_mat)

        center_mask = np.zeros((n_samples,), dtype=bool)
        center_mask[self.cluster_centers_] = True
        non_center_mask = ~center_mask
        # print("center_mask",center_mask.shape, non_center_mask.shape, self.dist_mat.shape)

        dists_to_centers = self.dist_mat[np.ix_(non_center_mask, center_mask)]
        min_dists_to_centers = dists_to_centers.min(axis=1)
        max_min_idx = np.argmax(min_dists_to_centers)
        non_center_indices = np.where(non_center_mask)[0]
        max_idx_absolute = non_center_indices[max_min_idx]

        # check
        if min_dists_to_centers[max_min_idx] > self.threshold:
            return max_idx_absolute
        else:
            return None

    def fit(self, X: np.ndarray, dist_mat=None) -> "FarthestPointClustering":
        """
        Args:
            X: (n_samples, n_features)
        Returns:
            self
        """
        n_samples = X.shape[0]

        if n_samples == 0:
            self.cluster_centers_ = np.array([])
            self.labels_ = np.array([])
            self.n_clusters_ = 0
            return self

        if dist_mat is None:
            dist_mat = compute_pairwise_distances_batch(X)
            self.dist_mat = dist_mat
        else:
            self.dist_mat = dist_mat

        self.max_distance = self.dist_mat.max()

        print("Find initial center...")
        idx1, idx2, max_distance = self._find_farthest_pair()

        if max_distance < self.threshold:
            print(f"Warning: max distance ({max_distance:.4f}) less than threshold ({self.threshold})")
            print("All points will be clustered to same center")
            self.cluster_centers_ = [idx1]
            self.n_clusters_ = 1
        else:
            self.cluster_centers_ = [idx1, idx2]
            print(f"Initial pair: {idx1}, {idx2}, Distance: {max_distance:.4f}")

        print("Find more cluster centers...")
        for i in range(n_samples - 2):
            next_center = self._find_next_center()
            if next_center is None:
                break
            self.cluster_centers_.append(next_center)

        self.n_clusters_ = len(self.cluster_centers_)
        print(f"Found {self.n_clusters_} cluster centers.")

        print("Assining points to centers...")
        self.labels_ = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            distances = [self.dist_mat[i, c] for c in self.cluster_centers_]
            self.labels_[i] = np.argmin(distances)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_features)
        Returns:
            np.ndarray:
        """
        if self.cluster_centers_ is None:
            raise ValueError("must first call fit()")

        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            distances = [np.linalg.norm(X[i] - self.get_centers()[c]) for c in range(self.n_clusters_)]
            labels[i] = np.argmin(distances)

        return labels

    def get_centers(self) -> np.ndarray:
        if self.cluster_centers_ is None:
            return np.array([])
        return np.array(self.cluster_centers_)

    def get_cluster_labels(self) -> np.ndarray:
        return self.labels_
