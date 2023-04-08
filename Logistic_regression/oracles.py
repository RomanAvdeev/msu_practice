import numpy as np
import scipy as sc
from scipy.special import expit


class BaseSmoothOracle:
    def func(self, X, y, w):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        if isinstance(X, sc.sparse.csr_matrix):
            X = X.toarray()
        return np.mean(np.logaddexp(0, - y * np.dot(X, w))) + self.l2_coef/2 * (np.linalg.norm(w) ** 2)

    def grad(self, X, y, w):
        return np.mean(((expit(y * np.dot(X, w)) - 1)[:, None] * y.reshape(len(y), 1) * X), axis=0) + self.l2_coef * w
