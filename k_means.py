import numpy as np

class MyCluster():

    def __init__(self, num_clusters=5, max_iter=100, balanced=False):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.balanced = balanced
        self.cluster_centers = []

    def fit(self, data_points):
        data_set = np.array(data_points)
        num_instances, num_features = np.shape(data_set)
        min_feature = np.min(data_set, axis=0)
        max_feature = np.max(data_set, axis=0)

        for _ in range(self.num_clusters):
            random_point = np.random.uniform(min_feature, max_feature)
            self.cluster_centers.append(random_point)

        clusters = np.zeros(num_instances, dtype=int)

        for _ in range(self.max_iter):
            for i, instance in enumerate(data_points):
                new_distances = self.calculate_distances(instance)
                cluster = np.argmin(new_distances)
                clusters[i] = cluster

            if self.balanced:
                clusters = self.balance_clusters(clusters, num_instances, data_points)

            new_centers = np.zeros((self.num_clusters, num_features))

            for i in range(self.num_clusters):
                instances_in_cluster = self.find_instances_in_cluster(i, data_set, clusters)
                if instances_in_cluster.any():
                    new_centers[i] = np.mean(instances_in_cluster, axis=0)
                else:
                    new_centers[i] = np.random.uniform(min_feature, max_feature)

            if np.all(new_centers == self.cluster_centers):
                break
            self.cluster_centers = new_centers

        return clusters.tolist(), self.cluster_centers.tolist()

    def calculate_distances(self, instance):
        distances = []
        for center in self.cluster_centers:
            distances.append(np.sqrt(np.sum(np.power(instance - center, 2))))
        return np.array(distances)

    def find_instances_in_cluster(self, index, data_set, clusters):
        instances = []
        for i in range(len(clusters)):
            if clusters[i] == index:
                instances.append(data_set[i])
        return np.array(instances)

    def balance_clusters(self, clusters, num_instances, data_points):
        size = num_instances // self.num_clusters
        for i in range(self.num_clusters):
            indices = np.where(clusters == i)[0]
            if len(indices) > size:
                num_change_needed = len(indices) - size
                for j in indices[:num_change_needed]:
                    distances = self.calculate_distances(data_points[j])
                    distances[i] = np.inf
                    clusters[j] = np.argmin(distances)
        return clusters
