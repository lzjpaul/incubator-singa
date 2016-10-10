import numpy as np
import warnings

from scipy import special, stats
from scipy.sparse import issparse

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import norm, safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
# from sklearn.base import SelectorMixin
from sklearn.feature_selection import SelectKBest
from sklearn.utils import (column_or_1d, as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)
import sys

def _chisquare(f_obs, f_exp):
    """Fast replacement for scipy.stats.chisquare.
    Version from https://github.com/scipy/scipy/pull/2525 with additional
    optimizations.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)

    print "f_obs = \n", f_obs
    print "f_exp = \n", f_exp
    k = len(f_obs)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    chisq /= f_exp
    chisq = chisq.sum(axis=0)
    print "chisq = \n", chisq[0:100]
    return chisq, special.chdtrc(k - 1, chisq)


def chi2(X, y):
    # XXX: we might want to do some of the following in logspace instead for
    # numerical stability.
    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features
    print "Y shape = \n", Y.shape
    print "X = \n", X
    print "observed = \n", observed

    feature_count = X.sum(axis=0).reshape(1, -1)
    print "feature_count = \n", feature_count
    class_prob = Y.mean(axis=0).reshape(1, -1)
    print "class_prob = \n", class_prob
    expected = np.dot(class_prob.T, feature_count)
    print "expected = \n", expected[0:100]
    return _chisquare(observed, expected)

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read all data
X_data = readData(sys.argv[1])  #modify here
X = np.array(X_data[0:])[:,0:]
X = X.astype(np.float)
#X = X[:,186:188]
X = X[:,42:44]

y_data = readData(sys.argv[2])  #modify here
y = np.array(y_data[0:])[:,0:]
y = y.astype(np.int)
y = column_or_1d(y, warn=True)

selected_fea_num = int(sys.argv[3])

print "X.shape = \n", X.shape
print "y.shape = \n", y.shape
print "chi2 value = \n", chi2(X, y)[0][0:100]

print "occurrence matrix, features only record appear or not"
for i in range(len(X[:,0])):
    for j in range(len(X[0,:])):
        if X[i][j] > float(0):
            X[i][j] = float(1)

print "max(X[0,:]) = \n", max(X[0,:])
print "max(X[100,:]) = \n", max(X[100,:])
print "max(X[200,:]) = \n", max(X[200,:])
print "max(X[1000,:]) = \n", max(X[1000,:])

print "X.shape = \n", X.shape
print "y.shape = \n", y.shape
# print "chi2 value = \n", chi2(X, y)[0][0:100]
# print "selected features = \n", X_new.get_support(indices=True)
# python chi2.py /ssd/zhaojing/cnn/NUHALLCOND/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_data_1case.csv /ssd/zhaojing/cnn/NUHALLCOND/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_label_1case.csv 20
