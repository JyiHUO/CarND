from scipy.sparse import coo_matrix
from sklearn.utils import resample
import numpy as np
import pandas as pd
def resample_equal_prob(X, y):
    # resample data for each class, so that they have equal prob
    standar = np.max(pd.value_counts(y))
    _, H, W, C = X.shape
    X_temp = X.reshape(X.shape[0], -1)
    y_temp = y[:, None]
    data = pd.DataFrame(np.concatenate([X_temp, y_temp], axis=1))
    res_x = []
    res_y = []

    y_mark = X_temp.shape[1]
    X_mark = list(range(X_temp.shape[1]))
    for i in np.unique(y_temp):
        X_temp = data[data[y_mark] == i][X_mark].values
        y_temp = data[data[y_mark] == i][y_mark].values
        X_sparse = coo_matrix(X_temp)
        n_sample = standar - X_temp.shape[0]
        if n_sample == 0:
            continue
        X_temp_, X_sparse_temp, y_temp_ = resample(X_temp, X_sparse, y_temp, n_samples=n_sample, random_state=0)
        X_temp = np.concatenate([X_temp_, X_temp], axis=0)
        y_temp = np.concatenate([y_temp_, y_temp])
        res_x.append(X_temp)
        res_y.append(y_temp)
    return np.concatenate(res_x, axis=0).reshape(-1, H, W, C), np.concatenate(res_y)