from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd
import sys
import os
import time
from math import log


# @ merge a matrix with discrete variables in column to a new discrete variable
def merge_matrix(mat_X):
    assert mat_X.size != 0

    new_x, idx = np.unique(mat_X, axis=0, return_index=True)
    n_es = new_x.shape[0]
    n_fs = new_x.shape[1]
    new_vec = np.array(
        [np.where((new_x == x.repeat(n_es).reshape(-1, n_es).T).sum(axis=1) == n_fs)[0] for x in mat_X]).flatten()

    return new_vec


# @ merge two discrete variables to a new discrete variable
def merge_two_variables(a, b):
    assert a.shape == b.shape

    c = np.vstack((a, b)).T   # row vectors stack, convert to column vectors
    unique_c, idx = np.unique(c, axis=0, return_index=True)
    new_vec = np.array([np.where((x[0] == unique_c[:, 0]) & (x[1] == unique_c[:, 1])) for x in c]).flatten()

    return new_vec


# maximum decision entropy of a discrete random variable
# ent = - {max_pi * log(max_pi, base=2) + (m - 1) (1 - max_pi) / (m - 1) *log((1 - max_pi) / (m - 1), base=2)}
def vector_entropy(vec, alpha=0.5, base=2):
    assert vec.ndim == 1   # vector only

    vec_len = len(vec)
    unique_v = np.unique(vec)
    count = np.bincount(vec)
    max_fre = np.max(count)   # maximum frequency

    max_pi = max_fre * 1.0 / vec_len

    if max_pi >= 1.0:
        ent = 0
    else:
        # print(max_pi, len(unique_v))
        ent = - alpha * max_pi * log(max_pi, base) \
              - (1 - alpha) * (1 - max_pi) * log((1 - max_pi) / (len(unique_v) - 1), base)

    return ent


# maximum decision entropy between two discrete random variables
# MH(D|C) = -sum_{Ci}[P(Ci) * [MP(D|Ci)logMP(D|Ci) + (1 - MP(D|Ci))log((1 - MP(D|Ci)) / (m -1))]
def maximum_decision_entropy(x, y, rows_n, alpha=0.5, base=2):
    assert x.shape == y.shape

    # compute the equivalence classes of x
    unique_x = np.unique(x)

    mde = 0
    all_indices = []
    for i in range(0, len(unique_x)):
        # get the indices of each condition equivalence class
        indices = np.where(x == unique_x[i])[0]
        dec_eqv_class = y[indices]
        H_Ci = vector_entropy(dec_eqv_class, alpha)
        mde = mde + float(len(dec_eqv_class)) / float(rows_n) * H_Ci

        # for accelerator
        if H_Ci <= 0:
            all_indices = np.hstack((all_indices, indices))

    # truncate the float number to avoid precision error
    mde = round(mde, 10)

    return mde, all_indices


# feature selection based on maximum decision entropy
def mde_fs(X, y, alpha=0.5):
    n_rows, n_cols = X.shape
    print(X.shape)

    # all features to one vector
    vector_all = merge_matrix(X)

    # calculate granular conditional entropy between all features and target class
    all_ce, _ = maximum_decision_entropy(vector_all, y, n_rows, alpha)

    # calculate gce between each feature and target class
    fea_ce = np.zeros(n_cols)
    for i in range(n_cols):
        f = X[:, i]
        fea_ce[i], _ = maximum_decision_entropy(f, y, n_rows, alpha)

    cur_ce, temp_ce, idx, err_thres = 0, 0, 0, 1E-12
    F = []         # index of selected features
    feas_ce = []   # gce between iteratively selected features and decision class: relevance
    f_select = []  # data with all selected features in one vector

    while True:
        if len(F) == 0:
            # first time selection: select the feature whose gce is the largest
            idx = np.argmin(fea_ce)
            F.append(idx)
            cur_ce = fea_ce[idx]
            feas_ce.append(cur_ce)
            f_select = X[:, idx]
            # print(fea_ce)

        if abs(all_ce - cur_ce) < err_thres:
            break

        cur_ce = 1E8
        for i in range(n_cols):
            if i not in F:
                f = X[:, i]
                new_f = merge_two_variables(f_select, f)
                temp_ce, _ = maximum_decision_entropy(new_f, y, n_rows, alpha)
                # print(i, temp_ce)

                # select the smallest mde and the corresponding feature index
                if temp_ce < cur_ce:
                    cur_ce = temp_ce
                    idx = i

        F.append(idx)
        feas_ce.append(cur_ce)
        f_select = merge_two_variables(f_select, X[:, idx])  # covert all selected feature vectors to one vector
        # print('Selected: ', idx, cur_ce)

    return F, feas_ce


def test_FS():
    data_name = 'example.data'
    file_path = '../data_PMDE/' + data_name

    save_path = file_path.replace('.data', '.txt')

    # **********01: load data**********
    data = np.loadtxt(file_path, dtype=int, delimiter=',')
    (_, m_features) = data.shape

    # shuffle the examples
    np.random.seed(1)
    np.random.shuffle(data)

    m_features = m_features - 1
    X = data[:, 0:m_features]
    y = data[:, -1]
    print("01:loading data: ", data.shape)

    # **********02: feature selection**********
    alpha = 0.5

    t_begin = time.time()
    feas_sel_mde, mde_val = mde_fs(X, y, alpha)
    t_end = time.time()
    print("\nElapsed Time: ", t_end - t_begin)
    reduce_rate = (m_features - len(feas_sel_mde)) / m_features
    print("\n01:all features:", m_features, ", mde selected:", len(feas_sel_mde), ", reduction rate:", reduce_rate)
    print(feas_sel_mde)
    print(mde_val)

    # **********03: cross-validation for performance evaluation**********
    clf = KNeighborsClassifier(n_neighbors=3)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    # obtain the dataset on the selected features
    features = X[:, feas_sel_mde]
    print(features.shape)

    scores = cross_val_score(clf, features, y, cv=kf)
    print("\n02:Performance with mde: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print(scores)

    print("All finished!")


if __name__ == '__main__':
    # test parameterized maximum decision entropy
    test_FS()
    # data_sets_FS_performance()




