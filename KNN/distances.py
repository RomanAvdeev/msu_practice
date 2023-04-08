import numpy as np


def euclidean_distance(X, Y):
    return np.sqrt(np.sum(np.square(Y), axis=1) - 2 * X @ Y.T + np.reshape(np.sum(np.square(X), axis=1),
                                                                           (X.shape[0], 1)))


def cosine_distance(X, Y):
    norm_X = np.sqrt(np.sum(np.square(X), axis=1))
    norm_X = np.where(norm_X != 0, norm_X, np.nan)
    norm_Y = np.sqrt(np.sum(np.square(Y), axis=1))
    norm_Y = np.where(norm_Y != 0, norm_Y, np.nan)
    res = np.divide(X@Y.T, norm_Y)
    res = np.divide(res, np.reshape(norm_X, (X.shape[0], 1)))
    return np.where(np.isnan(res), 0, 1 - res)
