import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy import sparse
def testaccuracy(w, v, x_test, y_test, clf_name):
    # w = w.toarray().reshape((1, x_test.shape[1]))
    # w = sparse.csr_matrix(w)
    # v = v.toarray().reshape((1, x_test.shape[1]))
    # v = sparse.csr_matrix(v)

    if clf_name == 'huber':
        # print "huber test add w and v"
        coef = np.add(w, v)
    else:
        # print "non-huber"
    	coef = w
    scores = x_test.dot(coef.T)
    scores = scores.toarray()
    scores = 1. / (1. + np.exp(-scores))

    threshold = .5
    y_pred = np.array(scores > threshold, dtype = np.int)

    y_test = y_test.toarray()
    if min(y_test) == -1:
	y_pred = 2. * y_pred - 1

    print "test y_test shape: ", y_test.shape
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = roc_auc_score(y_test, scores)
    print "test score: ", accuracy
    return accuracy
