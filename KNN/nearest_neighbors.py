from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance
import numpy as np


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size=10):
        self.k = k
        self.X = None
        self.y = None
        self.model = None
        self.test_block_size = test_block_size
        self.weights = weights
        self.metric = metric
        self.strategy = strategy

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.strategy == 'brute':
            self.model = NearestNeighbors(n_neighbors=self.k, algorithm='brute', metric=self.metric)
            self.model.fit(self.X, self.y)
        elif self.strategy == 'kd_tree':
            self.model = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree', metric=self.metric)
            self.model.fit(self.X, self.y)
        elif self.strategy == 'ball_tree':
            self.model = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', metric=self.metric)
            self.model.fit(self.X, self.y)

    def find_kneighbors(self, X, return_distance):
        if self.strategy == 'my_own':
            distance = np.zeros((X.shape[0], self.X.shape[0]))
            if self.metric == 'euclidean':
                distance = euclidean_distance(X, self.X)
            elif self.metric == 'cosine':
                distance = cosine_distance(X, self.X)

            indexes = np.argpartition(distance, self.k, axis=1)
            k_indexes = indexes[:, :self.k]
            index_row_k, index_row = np.indices((X.shape[0], self.k))
            nearest_k = distance[index_row_k, k_indexes]
            ind = np.arange(nearest_k.shape[0])[:, None], nearest_k.argsort(1)

            if return_distance:
                return nearest_k[ind], k_indexes[ind]
            else:
                return k_indexes[ind]
        else:
            return self.model.kneighbors(n_neighbors=self.k, X=X, return_distance=return_distance)

    def predict(self, X):
        distances, indexes = self.find_kneighbors(X, True)
        predicts = np.empty(X.shape[0])
        if self.weights:
            weight = 1 / (distances + 1e-5)
            for i, neighbor in enumerate(self.y[indexes]):
                predicts[i] = np.bincount(np.array(neighbor, dtype=int), weights=weight[i],
                                          minlength=len(np.unique(self.y[indexes]))).argmax()
        else:
            for i, neighbor in enumerate(self.y[indexes]):
                predicts[i] = np.bincount(np.array(neighbor, dtype=int),
                                          minlength=len(np.unique(self.y[indexes]))).argmax()
        return predicts
