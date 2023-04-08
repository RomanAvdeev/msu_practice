import numpy as np


def grad_finite_diff(function, w, eps=1e-3, **kwargs):
    grad = np.zeros(len(w))
    for i in range(len(w)):
        w_i = np.copy(w)
        w_i[i] += eps
        grad[i] = (function(w=w_i, **kwargs) - function(w=w, **kwargs)) / eps
    return grad
