import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist


class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
        return self

    def predict(self, X):
        predictions = None
        if self.X_train is not None and self.y_train is not None:
            dists = cdist(X, self.X_train, metric='euclidean')
            # get the indices of the k nearest neighbors
            k_indices = np.argpartition(dists, self.n_neighbors, axis=1)[
                :, :self.n_neighbors]

            # get the labels of the k nearest neighbors
            k_labels = self.y_train[k_indices]

            predictions = np.sign(np.sum(k_labels))
        return predictions
