import numpy as np
import random

class Cluster:

    def __init__(self, k_clusters=5, max_iterations=100, balanced=False):
        self.k_clusters = k_clusters
        self.max_iterations = max_iterations
        self.balanced = balanced
        self.centroids = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k_clusters, replace=False)] 
        labels = []
        for _ in range(self.max_iterations):
            labels = self._assign_labels(X)
            
            if self.balanced:
                labels = self._balance_clusters(X, labels)
                
            self._update_centroids(X, labels)
        return labels, self.centroids

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _balance_clusters(self, X, labels):
        sizes = np.bincount(labels)
        
        min_size = np.min(sizes)
        max_size = np.max(sizes)
        
        for _ in range(max_size - min_size):
            max_cluster = np.argmax(sizes)
            min_cluster = np.argmin(sizes)
            
            point_indices = np.where(labels == max_cluster)[0]
            i = random.choice(point_indices)
            
            labels[i] = min_cluster
            
            sizes[max_cluster] -= 1
            sizes[min_cluster] += 1
            
        return labels

    def _update_centroids(self, X, labels):
        for i in range(self.k_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)