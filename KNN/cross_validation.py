import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    res = []
    indexes = np.arange(n)
    folds = np.array(np.array_split(indexes, n_folds))
    for num in range(n_folds):
        arr_test = np.array(folds[num])
        elements = np.arange(n)
        arr_train = elements[np.isin(elements, arr_test, invert=True)]
        res.append((arr_train, arr_test))
    return res


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 5)
    res = {}
    for k_value in k_list:
        res[k_value] = np.empty(len(cv))
    j = 0
    for train_i, test_i in cv:
        X_train = X[train_i]
        y_train = y[train_i]
        X_test = X[test_i]
        y_test = y[test_i]
        classifier = KNNClassifier(k=max(k_list), **kwargs)
        classifier.fit(X_train, y_train)
        distances, indexes = classifier.find_kneighbors(X_test, return_distance=True)

        for k_value in k_list:
            curr_distances = distances[:, :k_value]
            curr_indexes = indexes[:, :k_value]

            train_marks = y_train[curr_indexes]
            predicts = np.empty(curr_indexes.shape[0])
            if classifier.weights:
                weight = 1 / (curr_distances + 1e-5)
                for i in range(predicts.size):
                    predicts[i] = np.argmax(np.bincount(train_marks[i], weight[i]))
            else:
                for i in range(curr_indexes.shape[0]):
                    predicts[i] = np.argmax(np.bincount(train_marks[i]))
            y_pred = predicts
            res[k_value][j] = np.mean(y_pred == y_test)
        j += 1
    return res