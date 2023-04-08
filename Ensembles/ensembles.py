import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        if feature_subsample_size is None:
            feature_subsample_size = 0.33
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.X = None
        self.y = None
        self.trees = []
        self.val_mse = None

    def bootstrap(self, X, y):
        bstraps = {}
        indexes = [i for i in range(X.shape[0])]
        for tree in range(self.n_estimators):
            tree_X = X[indexes, :]
            tree_y = y[indexes]
            bstraps[tree] = {'boot': tree_X, 'y': tree_y}
        return bstraps

    def fit(self, X, y, X_val=None, y_val=None):
        val_check = False
        errors = np.zeros((self.n_estimators, 1))

        if X_val is not None and y_val is not None:
            val_check = True
            errors = np.zeros((self.n_estimators, X_val.shape[0]))

        bstraps = self.bootstrap(X, y)
        for n in range(self.n_estimators):
            tree = DecisionTreeRegressor(criterion='squared_error', max_depth=self.max_depth,
                                         max_features=int(np.ceil(X.shape[1] * self.feature_subsample_size)),
                                         **self.trees_parameters)

            tree.fit(bstraps[n]['boot'], bstraps[n]['y'])
            self.trees.append(tree)
            if val_check:
                errors[n, :] = mean_squared_error(y_val, tree.predict(X_val))
        if val_check:
            self.val_mse = np.mean(errors, axis=0)

    def predict(self, X):
        if self.trees is not None:
            preds = np.zeros((self.n_estimators, X.shape[0]))
            for n in range(len(self.trees)):
                tree_y_pred = self.trees[n].predict(X)
                preds[n, :] = tree_y_pred
            return np.mean(preds, axis=0)
        else:
            return None


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        if feature_subsample_size is None:
            feature_subsample_size = 0.33
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []
        self.weights = None
        self.val_mse = []

    def bootstrap(self, X, y):
        bstraps = {}
        indexes = [i for i in range(X.shape[0])]
        for tree in range(self.n_estimators):
            tree_X = X[indexes, :]
            tree_y = y[indexes]
            bstraps[tree] = {'boot': tree_X, 'y': tree_y}
        return bstraps

    def fit(self, X, y, X_val=None, y_val=None):
        """
            X : numpy ndarray
                Array of size n_objects, n_features

            y : numpy ndarray
                Array of size n_objects
        """
        bstraps = self.bootstrap(X, y)

        val_check = False
        val_preds = np.zeros(1)

        if X_val is not None and y_val is not None:
            val_check = True
            val_preds = np.zeros(X_val.shape[0])

        shift = y.copy()
        preds = np.zeros(X.shape[0])

        self.weights = [1]

        for n in range(self.n_estimators):
            tree = DecisionTreeRegressor(criterion='squared_error', max_depth=self.max_depth,
                                         max_features=int(np.ceil(X.shape[1] * self.feature_subsample_size)),
                                         **self.trees_parameters)
            tree.fit(bstraps[n]['boot'], shift)
            self.trees.append(tree)

            predict = tree.predict(bstraps[n]['boot'])
            w = minimize_scalar(lambda x: np.sum((x * predict - shift) ** 2)).x
            self.weights.append(w)

            preds += self.learning_rate * self.weights[n] * predict
            shift = y - preds

            if val_check:
                val_preds += self.learning_rate * self.weights[n] * tree.predict(X_val)
                self.val_mse.append(np.sqrt(np.mean((y_val - val_preds) ** 2)))

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if self.trees is not None:
            preds = np.zeros(X.shape[0])
            for n in range(self.n_estimators):
                preds += self.learning_rate * self.weights[n] * self.trees[n].predict(X)
            return preds
        else:
            return None
