import numpy as np
from oracles import BinaryLogistic
import time
from scipy.special import expit


class GDClassifier:
    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        self.X = None
        self.y = None
        self.w = None
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.my_oracle = None
        if loss_function == 'binary_logistic':
            self.my_oracle = BinaryLogistic(**kwargs)

    def fit(self, X, y, w_0=None, trace=False):
        self.X = X
        self.y = y
        self.w = w_0

        gradient = self.my_oracle.grad(self.X, self.y, self.w)
        history = {'time': [], 'func': [], 'weights': []}
        pred_func = 0
        time_step = -1

        for k in range(1, self.max_iter + 1):
            if time_step != -1:
                time_res = time.time() - time_step
                history['time'].append(time_res)
                time_step = time.time()
            else:
                time_step = time.time()
            func = self.my_oracle.func(self.X, self.y, self.w)
            history['func'].append(round(func, 12))
            history['weights'].append(self.w)
            if abs(func - pred_func) >= self.tolerance:
                learning_rate = self.step_alpha / (k ** self.step_beta)
                self.w = self.w - learning_rate * gradient
                gradient = self.my_oracle.grad(self.X, self.y, self.w)
                pred_func = func
            else:
                break
        if trace is True:
            return history

    def predict(self, X):
        return np.sign(X @ self.w)

    def predict_proba(self, X):
        probabilities = np.array((X.shape[0], 2))
        return expit(X @ self.w)

    def get_objective(self, X, y):
        return float(self.my_oracle.func(X, y, self.w))

    def get_gradient(self, X, y):
        return np.array(self.my_oracle.grad(X, y, self.w))

    def get_weights(self):
        return self.w


class SGDClassifier(GDClassifier):
    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        self.random_seed = random_seed
        self.batch_size = batch_size
        super().__init__(loss_function, step_alpha=1, step_beta=0, tolerance=1e-5,
                         max_iter=1000, **kwargs)

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        np.random.seed(self.random_seed)
        self.X = X
        self.y = y
        self.w = w_0

        indexes = np.random.choice(self.X.shape[0], self.batch_size, replace=False)
        gradient = self.my_oracle.grad(self.X[indexes, :], self.y[indexes], self.w)
        history = {'epoch_num': [], 'time': [], 'func': [], 'weights_diff': []}
        pred_func = 0
        time_step = -1

        epoch_num, pred_epoch_num, analyzed_elems = 0, 0, 0
        pred_weights = self.w
        for k in range(1, self.max_iter + 1):
            epoch_num = analyzed_elems / self.X.shape[0]
            func = self.my_oracle.func(self.X, self.y, self.w)

            if epoch_num == 0:
                time_step = time.time()

            if abs(epoch_num - pred_epoch_num) > log_freq:
                pred_epoch_num = epoch_num
                time_res = time.time() - time_step
                history['time'].append(time_res)
                history['func'].append(round(func, 12))
                history['weights'].append(self.w)
                history['epoch_num'].append(epoch_num)
                history['weights_diff'].append(np.linalg.norm(self.w - pred_weights) ** 2)

            if abs(func - pred_func) >= self.tolerance:
                learning_rate = self.step_alpha / (k ** self.step_beta)
                self.w = self.w - learning_rate * gradient
                indexes = np.random.choice(self.X.shape[0], self.batch_size, replace=False)
                gradient = self.my_oracle.grad(self.X[indexes, :], self.y[indexes], self.w)
                pred_func = func
            else:
                break
            analyzed_elems += self.batch_size

        if trace is True:
            return history
