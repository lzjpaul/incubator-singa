# _7: no adaptive + kmeans decide which gaussian to belong to
# _5: no adaptive
# (0) overall logic
# (1) better coding method?
# (2) correct? compared to dense
# (3) all are csr matrix?
# sparse: w + v
# sparse: linalg.norm(no ord)
# ?? batchsize * batch_iter / shape[0]?
# gradient descent: divided by # of samples --> float?
__author__ = "Luo Zhaojing"
__version__ = "0.1.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
import random
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from testdata import testaccuracy
from LDA_Gibbs import LdaSampler
from sklearn.cluster import KMeans
import collections
def huber_grad_descent_avg(batch_X, batch_y, w, v, param, C, is_l1):
    # sparse: if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in huber gd avg"
    # print "batch_X norm: ", np.linalg.norm(batch_X)
    # print "w shape: ", w.shape
    # print "batch_X shape: ", batch_X.shape
    batch_y = batch_y.T
    # print "batch_y shape: ", batch_y.shape
    # print "(-batch_y * (w + v).dot(batch_X.T) shape: ", (-batch_y * (w + v).dot(batch_X.T)).shape
    grad = param * v.sign() if is_l1 else param * 2.0 * w
    grad = grad.toarray()
    grad[0, -1] = 0.0
    grad = sparse.csr_matrix(grad)
    # print "grad shape1: ", grad.shape
    # print "-batch_y shape: ",  (-batch_y).shape
    # print "(w + v) shape: ", (w + v).shape
    # print "((w + v).dot(batch_X.T)) shape: ", ((w + v).dot(batch_X.T)).shape
    # print "finish"
    f1 = np.exp(((-batch_y).multiply((w + v).dot(batch_X.T))).toarray())
    # print "f1 shape: ", f1.shape
    # print "batch_X.T shape: ", batch_X.T.shape
    # print "batch_y shape", batch_y.shape
    # print "(f1 / (1.0 + f1)) shape: ", (f1 / (1.0 + f1)).shape
    # print "(C * (-1 * batch_y.T).toarray() shape: ", (C * -batch_y).toarray().shape
    # print "(C * ((-batch_y).multiply(f1 / (1.0 + f1)))) shape: ", (C * ((-batch_y).multiply(f1 / (1.0 + f1)))).shape
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    # print "y_res shape: ", y_res.shape
    # print "f1 shape: ", f1.shape
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    # y_res = y_res.flatten()
    # print "y_res shape: ", y_res.shape
    # print "y_res: ", y_res
    # print "isspmatrix_csr(y_res): ", sparse.isspmatrix_csr(y_res)
    # print "issparse(y_res): ", sparse.issparse(y_res)
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    # print "res shape: ", res.shape
    res = res.T
    # print "isspmatrix_csr(res): ", sparse.isspmatrix_csr(res)
    #print "issparse(res): ", sparse.issparse(res)
    ressum = res.sum(axis=0) #this will turn back to dense
    # print "isspmatrix_csr(ressum): ", sparse.isspmatrix_csr(ressum)
    # print "issparse(ressum): ", sparse.issparse(ressum)
    ressum = ressum.astype(np.float)
    # print "isspmatrix_csr(ressum): ", sparse.isspmatrix_csr(ressum)
    ressum /=  float(batch_X.shape[0])
    # print "grad shape: ", grad.shape
    # print "ressum shape: ", ressum.shape
    # print "isspmatrix_csr(grad + ressum)", sparse.isspmatrix_csr(grad + ressum)
    # print "grad + sparse.csr_matrix(ressum) norm: ", linalg.norm(grad + sparse.csr_matrix(ressum))
    return grad + sparse.csr_matrix(ressum)

def gaussian_mixture_descent_avg(batch_X, batch_y, w, theta_vec, lambda_vec, C):
    #if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in lasso gd avg"
    batch_y = batch_y.T
    w_array = w.toarray() # in order for np.exp
    # print "w shape: ", w.shape
    w_array = np.reshape(w_array, w_array.shape[1])
    # grad = param * w.sign()
    # print "lasso w shape: ", w.shape
    grad_denominator = np.zeros(w_array.shape[0]-1)
    grad_numerator = np.zeros(w_array.shape[0]-1)
    print "bias: ", w_array[-1]
    for i in range(theta_vec.shape[0]):
        print "gaussian theta i: ", i
        grad_denominator = grad_denominator + theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array[:-1] * w_array[:-1])
    for i in range(theta_vec.shape[0]):
        grad_numerator = grad_numerator + theta_vec[i] * np.power((lambda_vec[i]/ (2 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array[:-1] * w_array[:-1]) * lambda_vec[i] * w_array[:-1]
    grad = np.zeros(w_array.shape[0])
    grad[:-1] = grad_numerator / grad_denominator.astype(float) # -log(p(w))
    grad[-1] = 0.0
    print "grad[0:10]: ", grad[0:10]
    print "(lambda_vec[0] * w_array)[0:10]: ", (lambda_vec[0] * w_array)[0:10]
    # grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return sparse.csr_matrix(grad + ressum)

def gaussian_mixture_gd_em_descent_avg(batch_X, batch_y, res_matrix, w, theta_vec, lambda_vec, a, b, theta_alpha, C):
    #if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in lasso gd avg"
    print "lambda_vec shape: ", lambda_vec.shape
    print "theta_vec shape: ", theta_vec.shape
    batch_y = batch_y.T
    w_array = w.toarray() # in order for np.exp
    # print "w shape: ", w.shape
    w_array = np.reshape(w_array, w_array.shape[1])
    w_weight_array = w_array[:-1].reshape((-1, 1)) # no bias
    # grad = param * w.sign()
    # print "lasso w shape: ", w.shape
    print "bias: ", w_array[-1]
    lambda_w = np.dot(w_weight_array, lambda_vec.reshape((1, -1)))
    # print "lambda_w shape: ", lambda_w.shape
    grad = np.zeros(w_array.shape[0])
    grad[:-1] = np.sum((res_matrix * lambda_w), axis=1)# -log(p(w))
    grad[-1] = 0.0
    # print "grad[0:10]: ", grad[0:10]
    # print "(lambda_vec[0] * w_array)[0:10]: ", (lambda_vec[0] * w_array)[0:10]
    # grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])

    ###lambda_minimizer#########
    lambda_numerator = 0.5 * res_matrix
    lambda_numerator = np.sum(lambda_numerator, axis=0)
    print "lambda_numerator shape: ", lambda_numerator.shape
    lambda_numerator = lambda_numerator + a -1
    # print "w_weight_array shape: ", w_weight_array.shape
    # print "diag shape: ", np.diag(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])).shape
    # print "res_matrix.T shape: ", res_matrix.T.shape
    lambda_denominator = sparse.csr_matrix(res_matrix.T).dot(sparse.diags(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])))
    lambda_denominator = (lambda_denominator.toarray().T)
    lambda_denominator = np.sum(lambda_denominator, axis=0)
    lambda_denominator = lambda_denominator + b
    lambda_vec_minimizer = (lambda_numerator / lambda_denominator.astype(float))
    print "1/w^2: ", (0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0]))[:100]
    ###lambda_minimizer#########

    ###theta_minimizer#########
    theta_numerator = np.sum(res_matrix, axis=0) + theta_alpha - 1
    theta_denominator = np.sum(res_matrix) + theta_vec.shape[0] * (theta_alpha - 1)
    theta_vec_minimizer = (theta_numerator / theta_denominator.astype(float))
    ###theta_minimizer#########
    # print "in descent (grad + ressum)[:10]", (grad + ressum)[0, :10]
    # print "in descent theta_vec_minimizer: ", theta_vec_minimizer
    # print "in descent lambda_vec_minimizer: ", lambda_vec_minimizer
    return sparse.csr_matrix(grad + ressum), theta_vec_minimizer, lambda_vec_minimizer

def gaussian_mixture_gd_descent_avg(batch_X, batch_y, res_matrix, w, theta_r_vec, theta_vec, lambda_t_vec, lambda_vec, a, b, theta_alpha, C):
    #if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in lasso gd avg"
    batch_y = batch_y.T
    w_array = w.toarray() # in order for np.exp
    # print "w shape: ", w.shape
    w_array = np.reshape(w_array, w_array.shape[1])
    w_weight_array = w_array[:-1].reshape((-1, 1)) # no bias
    # grad = param * w.sign()
    # print "lasso w shape: ", w.shape
    # print "bias: ", w_array[-1]
    lambda_w = np.dot(w_weight_array, lambda_vec.reshape((1, -1)))
    # print "lambda_w shape: ", lambda_w.shape
    grad = np.zeros(w_array.shape[0])
    grad[:-1] = np.sum((res_matrix * lambda_w), axis=1)# -log(p(w))
    grad[-1] = 0.0
    # print "grad[0:10]: ", grad[0:10]
    # print "(lambda_vec[0] * w_array)[0:10]: ", (lambda_vec[0] * w_array)[0:10]
    # grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])

    ###lambda_t_update#########
    print "float(a-1): ", float(a-1)
    term1 = (float(a-1) / lambda_vec) - b
    print "term1: ", term1
    res_w = sparse.csr_matrix(res_matrix.T).dot(sparse.diags(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])))
    res_w = (res_w.toarray().T)
    term2 = np.sum((res_matrix / (2.0 * lambda_vec.reshape(1, -1))) - res_w, axis=0)
    # print "gd lambda_t term2: ", term2[:100]
    lambda_t_vec_update = (- term1 - term2) * lambda_vec # lambda = exp(lambda_t)derivative of lambda to t
    # lambda_t_vec_update = (- term1 - term2) * (-lambda_vec) # should change mapping as well!!! lambda^(-1) = exp(lambda_t)derivative of lambda to t
    ###lambda_t_update#########

    ###theta_r_update#########
    theta_k_r_j = np.zeros((theta_vec.shape[0], theta_vec.shape[0])) ## derivative of theta_k to r_j
    for k in range(theta_k_r_j[:,0].shape[0]):
        for j in range(theta_k_r_j[0, :].shape[0]):
            if k == j:
                theta_k_r_j[k, j] = (theta_vec[k] - theta_vec[k] * theta_vec[k])
            else:
                theta_k_r_j[k, j] = (- theta_vec[k] * theta_vec[j])

    theta_r_vec_update = np.zeros(theta_r_vec.shape[0])
    term1 = (theta_alpha - 1) / theta_vec.astype(float)
    print "theta_vec.astype(float): ", theta_vec.astype(float)
    term2 = np.sum(res_matrix.astype(float) / (theta_vec.reshape(1, -1)), axis=0)
    theta_derivative = ( - term1 - term2)
    for j in range(theta_r_vec_update.shape[0]): #r_j
        theta_r_vec_update[j] = np.sum(theta_derivative * theta_k_r_j[:, j])
    # print "-term1: ", -term1
    # print "-term2: ", -term2
    # print "theta_derivative: ", theta_derivative
    # print "theta_k_r_j: ", theta_k_r_j
    # print "theta_r_vec_update: ", theta_r_vec_update


    ###theta_r_update#########
    return sparse.csr_matrix(grad + ressum), theta_r_vec_update, lambda_t_vec_update

def lasso_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    #if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in lasso gd avg"
    batch_y = batch_y.T
    grad = param * w.sign()
    grad = grad.toarray()
    grad[0, -1] = 0.0
    grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape
    # print "lasso w shape: ", w.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + sparse.csr_matrix(ressum)

def ridge_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    # if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in ridge gd avg"
    batch_y = batch_y.T
    # print "param * 2.0: ", param * 2.0
    grad = param * 2.0 * w
    grad = grad.toarray()
    # print "before grad: ", grad[0,-2]
    grad[0, -1] = 0.0
    # print "after grad: ", grad[0, 0:10]
    # print "grad var: ", np.var(grad)
    # print "grad mean: ", np.mean(grad)
    grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + sparse.csr_matrix(ressum)

def elasticnet_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    # if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in elastic gd avg"
    batch_y = batch_y.T
    grad = param * l1_ratio_or_mu * w.sign() + param * (1 - l1_ratio_or_mu) * w
    grad = grad.toarray()
    grad[0, -1] = 0.0
    grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + sparse.csr_matrix(ressum)


def smoothing_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    # print "begin toarray"
    # if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "end toarray"
    # print "in smoothing gd avg"
    batch_y = batch_y.T
    w_array = w.toarray()
    # print "smoothing w_array shape: ", w_array.shape
    w_array_exp = np.exp(w_array)
    grad =  param * (w_array_exp - 1) / (w_array_exp + 1) # log(1+e^(-w)) + log(1+e^(w))
    # grad = grad.toarray()
    grad[0, -1] = 0.0
    # grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape
    # grad = param * w.sign()
    #print "not smoothing formula"
    # print "grad: ", grad
    # print "float(batch_X.shape[0]): ", float(batch_X.shape[0])
    # print 'grad.shape: ', grad.shape
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return sparse.csr_matrix(grad + ressum)

def huber_optimizator_avg(X_train, y_train, X_test, y_test, lambd, l1_ratio_or_mu, C, max_iter, eps, alpha, decay, batch_size, clf_name):
    k = 0
    w = np.zeros(X_train.shape[1])
    v = np.zeros(X_train.shape[1])
    w = sparse.csr_matrix(w)
    v = sparse.csr_matrix(v)
    y_train = sparse.csr_matrix(y_train)
    y_train = y_train.T
    y_test = sparse.csr_matrix(y_test)
    y_test = y_test.T
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)

    print "max_iter: ", max_iter
    accuracy = 0.0
    best_accuracy = 0.0
    best_accuracy_step = 0
    batch_iter = 0
    np.random.seed(10)
    idx = np.random.permutation(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]

    grad_descent_avg = huber_grad_descent_avg
    while True:
        index = (batch_size * batch_iter) % X_train.shape[0]

        if (index + batch_size) >= X_train.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

        batch_X, batch_y = X_train[index : (index + batch_size)], y_train[index : (index + batch_size)]

        vec = (w + v)
        # making optimization in w and v
        # w and v will be automatically changed to sparse array
        # print "v-="
        v -= alpha * grad_descent_avg(batch_X, batch_y, w, v, l1_ratio_or_mu, C, True)
        # print "isspmatrix_csr(v): ", sparse.isspmatrix_csr(v)
        # print "w-="
        w -= alpha * grad_descent_avg(batch_X, batch_y, w, v, lambd, C, False)
        # v_update = alpha * grad_descent_avg(batch_X, batch_y, w, v, l1_ratio_or_mu, C, True)
        # v -= v_update
        # print "v_update norm: ", np.linalg.norm(v_update)
        # w_update = alpha * grad_descent_avg(batch_X, batch_y, w, v, lambd, C, False)
        # w -= w_update
        # print "w_update norm: ", np.linalg.norm(w_update)

        alpha -= alpha * decay
        k += 1

        if k % 20 == 0: # 20
            print "huber_optimizator k: ", k
        if k % 60 == 0: # 100
            print "test at step: ", k
            accuracy = testaccuracy(w, v, X_test, y_test, 'huber')
            print "accuracy this step: ", accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_step = k
            print "best_accuracy: ", best_accuracy
            print "best_accuracy_step: ", best_accuracy_step


        batch_iter = batch_iter + 1
        # if k >= max_iter or linalg.norm(((w+v) - vec)) < eps:
        # if k >= max_iter:
        #     break
        if k >= max_iter or linalg.norm(((w+v) - vec)) < eps:
            break
        if np.isnan(linalg.norm(w)) or np.isinf(linalg.norm(w)):
            return k, w.toarray(), best_accuracy, -1
    print "huber opt avg final k: ", k
    return k, w.toarray(), v.toarray(), best_accuracy, best_accuracy_step

# only one weight w, not w + v
def non_huber_optimizator_avg(X_train, y_train, X_test, y_test, lambd, l1_ratio_or_mu, C, max_iter, eps, alpha, decay, batch_size, clf_name):
    k = 0
    w = np.zeros(X_train.shape[1])
    w = sparse.csr_matrix(w)
    y_train = sparse.csr_matrix(y_train)
    y_train = y_train.T
    y_test = sparse.csr_matrix(y_test)
    y_test = y_test.T
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    print "optimizator_avg lambd: ", lambd
    print "optimizator_avg alpha: ", alpha
    print "optimizator_avg max_iter: ", max_iter
    accuracy = 0.0
    best_accuracy = 0.0
    best_accuracy_step = 0
    batch_iter = 0
    np.random.seed(10)
    idx = np.random.permutation(X_train.shape[0])
    print "data idx: ", idx
    X_train = X_train[idx]
    y_train = y_train[idx]
    if clf_name == 'lasso':
        grad_descent_avg = lasso_grad_descent_avg
    elif clf_name == 'ridge':
        grad_descent_avg = ridge_grad_descent_avg
    elif clf_name == 'elasticnet':
        grad_descent_avg = elasticnet_grad_descent_avg
    elif clf_name == 'smoothing':
        grad_descent_avg = smoothing_grad_descent_avg
    while True:
        # sparse matrix works, random.shuffle
        # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
        # print "X.shape: ", X.shape
        # print "max idx: ", max(idx)
        index = (batch_size * batch_iter) % X_train.shape[0]

        if (index + batch_size) >= X_train.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

        batch_X, batch_y = X_train[index : (index + batch_size)], y_train[index : (index + batch_size)]
        w_update = alpha * grad_descent_avg(batch_X, batch_y, w, lambd, l1_ratio_or_mu, C)
        # print "w_update norm: ", linalg.norm(w_update)
        w -= w_update
        # print "w norm: ", linalg.norm(w)
        alpha -= alpha * decay
        k += 1

        if k % 20 == 0:
            print "non_huber_optimizator k: ", k
        if k % 60 == 0:
            print "test at step: ", k
            accuracy = testaccuracy(w, w, X_test, y_test, 'non-huber')
            print "accuracy this step: ", accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_step = k
            print "best_accuracy: ", best_accuracy
            print "best_accuracy_step: ", best_accuracy_step

        batch_iter = batch_iter + 1
        # if k >= max_iter or linalg.norm(w_update) < eps:
        # if k >= max_iter:
        #    break
        if k >= max_iter or linalg.norm(w_update) < eps:
            break
        if np.isnan(linalg.norm(w)) or np.isinf(linalg.norm(w)):
            return k, w.toarray(), best_accuracy, -1
    print "non_huber opt avg final k: ", k
    return k, w.toarray(), best_accuracy, best_accuracy_step

def gaussian_mixture_optimizator_avg(X_train, y_train, X_test, y_test, C, max_iter, eps, alpha, n_gaussian, theta_alpha, a, b, decay, batch_size, clf_name, batchgibbs, initial_L2_lambd, initial_L2_alpha, initial_L2_step):
    k = 0
    w = np.zeros(X_train.shape[1])
    w = sparse.csr_matrix(w)
    y_train = sparse.csr_matrix(y_train)
    y_train = y_train.T
    y_test = sparse.csr_matrix(y_test)
    y_test = y_test.T
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    print "max_iter: ", max_iter
    print "lerning rate alpha: ", alpha
    print "C: ", C
    accuracy = 0.0
    best_accuracy = 0.0
    best_accuracy_step = 0
    batch_iter = 0
    np.random.seed(10)
    idx = np.random.permutation(X_train.shape[0])
    print "data idx: ", idx
    X_train = X_train[idx]
    y_train = y_train[idx]
    # pre_w = np.copy(w.toarray()) #dense
    # print "pre_w.shape: ", pre_w.shape
    # pre_w = np.reshape(pre_w, pre_w.shape[1])
    print "in optimizator_avg n_gaussian: ", n_gaussian
    print "in optimizator_avg theta_alpha: ", theta_alpha
    print "in optimizator_avg a: ", a
    print "in optimizator_avg b: ", b
    # sampler = LdaSampler(n_gaussians=n_gaussian, alpha = (1.0 / n_gaussian), a = a, b = b) #number of gaussians
    sampler = LdaSampler(n_gaussians=n_gaussian, alpha = theta_alpha, a = a, b = b) #number of gaussians
    while True:
        # sparse matrix works, random.shuffle
        # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
        # print "X.shape: ", X.shape
        # print "max idx: ", max(idx)
        index = (batch_size * batch_iter) % X_train.shape[0]

        if (index + batch_size) >= X_train.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

        batch_X, batch_y = X_train[index : (index + batch_size)], y_train[index : (index + batch_size)]

        if k >= initial_L2_step:
            if k == initial_L2_step:
                print "w_update_hist shape: ", w_update_hist.shape
                w_weight_update_hist_var = np.var(w_update_hist, axis=0)[:-1] # the last is bias
                print "w_weight_update_hist_var shape: ", w_weight_update_hist_var.shape
                km = KMeans(n_clusters=n_gaussian)
                km.fit(w_weight_update_hist_var.reshape(-1,1))
                cluster_counts = collections.Counter(km.labels_)
                print "cluster_counts: ", cluster_counts
                print "cluster centers: ", km.cluster_centers_
                sorted_cluster_centers_index = np.argsort(-km.cluster_centers_.reshape(1,-1))
                sorted_cluster_centers_index = sorted_cluster_centers_index.reshape(sorted_cluster_centers_index.shape[1])
                theta_vec = np.zeros(n_gaussian)
                lambda_vec = np.zeros(n_gaussian)
                for i in range(n_gaussian): #cluster center value from large to small
                    theta_vec[i] = cluster_counts[sorted_cluster_centers_index[i]]
                    theta_vec[i] = theta_vec[i] / float(w_weight_update_hist_var.shape[0])
                lambda_vec[0] = 2
                for i in range(1, n_gaussian):
                    lambda_vec[i] = (lambda_vec[i-1] / 10.0)



            ############LDA_sampler#################
            print "before sampler w: ", linalg.norm(w)
            # theta_vec, lambda_vec = sampler.run(np.reshape((w.toarray()*np.sqrt(9375))[0,:-1], (w.toarray().shape[1]-1)), (k-initial_L2_step), batchgibbs)
            # print "theta_vec: ", theta_vec
            # print "lambda_vec: ", lambda_vec
            print "enforce theta_vec: ", theta_vec
            print "enforce lambda_vec: ", lambda_vec
            # lambda_vec[0] = 2
            # print "lambda_vec enforce: ", lambda_vec
            ############LDA_sampler#################
            w_update = alpha * gaussian_mixture_descent_avg(batch_X, batch_y, w, theta_vec, lambda_vec, C)
            # print "w_update norm: ", linalg.norm(w_update)
            w -= w_update
            # print "w: ", linalg.norm(w)
            # print "w: ", w
            alpha -= alpha * decay
            print "lr alpha minus two timessss: ", alpha
        else: #L2_norm
            # w_update = initial_L2_alpha * ridge_grad_descent_avg(batch_X, batch_y, w, initial_L2_lambd, 0.1, C)
            w_update = alpha * ridge_grad_descent_avg(batch_X, batch_y, w, initial_L2_lambd, 0.1, C)
            # print "w_update norm: ", linalg.norm(w_update)

            w -= w_update
            # print "w: ", linalg.norm(w)
            # print "w: ", w

           ####### kmeans: w_update history ########
            if k == 0:
                # w_update_hist = (w_update.toarray() / alpha).reshape(1, -1) #correct...?
                w_update_hist = (w.toarray()).reshape(1, -1) #correct...?
            else:
                # w_update_hist = np.concatenate((w_update_hist, (w_update.toarray() / alpha).reshape(1,-1)), axis=0)
                w_update_hist = np.concatenate((w_update_hist, w.toarray().reshape(1,-1)), axis=0)
            ###### kmeans: w_update history #######




            # initial_L2_alpha -= initial_L2_alpha * decay
            alpha -= alpha * decay
            print "lr alpha: ", alpha
        # w -= w_update
        # print "w: ", linalg.norm(w)
        # alpha -= alpha * decay
        k += 1

        if k % 20 == 0:
            print "non_huber_optimizator k: ", k
        if k % 60 == 0:
            print "test at step: ", k
            accuracy = testaccuracy(w, w, X_test, y_test, 'non-huber')
            print "accuracy this step: ", accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_step = k
            print "best_accuracy: ", best_accuracy
            print "best_accuracy_step: ", best_accuracy_step

        batch_iter = batch_iter + 1
        # if k >= max_iter or linalg.norm(w_update) < eps:
        # if k >= max_iter:
        # if k >= 20:
        #     break
        if k >= max_iter or linalg.norm(w_update) < eps:
            break
        if np.isnan(linalg.norm(w)) or np.isinf(linalg.norm(w)):
            return k, w.toarray(), best_accuracy, -1
    print "gaussian opt avg final k: ", k
    return k, w.toarray(), best_accuracy, best_accuracy_step

# only one weight w, not w + v
def gaussian_mixture_gd_em_optimizator_avg(X_train, y_train, X_test, y_test, C, max_iter, eps, alpha, n_gaussian, theta_alpha, a, b, decay, batch_size, clf_name):
    k = 0
    #w = np.zeros(X_train.shape[1])
    # w = np.random.normal(0.0, 0.1, X_train.shape[1])
    np.random.seed(10)
    w = np.random.normal(0.0, 0.001, X_train.shape[1])
    w = sparse.csr_matrix(w)
    y_train = sparse.csr_matrix(y_train)
    y_train = y_train.T
    y_test = sparse.csr_matrix(y_test)
    y_test = y_test.T
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    print "gaussian_mixture_gd_em_optimizator_avg max_iter: ", max_iter
    print "lerning rate alpha: ", alpha
    print "C: ", C
    accuracy = 0.0
    best_accuracy = 0.0
    best_accuracy_step = 0
    batch_iter = 0
    # np.random.seed(10)
    idx = np.random.permutation(X_train.shape[0])
    print "data idx: ", idx
    X_train = X_train[idx]
    y_train = y_train[idx]
    print "in optimizator_avg n_gaussian: ", n_gaussian
    print "in optimizator_avg theta_alpha: ", theta_alpha
    print "in optimizator_avg a: ", a
    print "in optimizator_avg b: ", b
    # initialization !!
    theta_vec = np.zeros(n_gaussian)
    for i in range(n_gaussian):
        theta_vec[i] = (1.0 / n_gaussian)
    # lambda_vec = np.random.uniform(0.0, 0.1, n_gaussian)
    lambda_vec = np.array([1/2., 1/4., 1/8.])
    print "theta_vec: ", theta_vec
    print "lambda_vec: ", lambda_vec
    while True:
        # sparse matrix works, random.shuffle
        # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
        # print "X.shape: ", X.shape
        # print "max idx: ", max(idx)
        index = (batch_size * batch_iter) % X_train.shape[0]

        if (index + batch_size) >= X_train.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

        batch_X, batch_y = X_train[index : (index + batch_size)], y_train[index : (index + batch_size)]

        ##update responsibility##
        print "theta_vec: ", theta_vec
        print "lambda_vec: ", lambda_vec
        w_array = w.toarray() # in order for np.exp
        w_array = np.reshape(w_array, w_array.shape[1])
        res_denominator = np.zeros(w_array.shape[0]-1)
        for i in range(theta_vec.shape[0]):
            # print "gaussian theta i: ", i
            res_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array[:-1] * w_array[:-1])
            # print "res_denominator_inc[0]: ", res_denominator_inc[0]
            # print "res_denominator_inc shape: ", res_denominator_inc.shape
            if i == 0:
                res_matrix = np.reshape(res_denominator_inc, (-1, 1))
            else:
                res_matrix = np.concatenate((res_matrix, np.reshape(res_denominator_inc, (-1, 1))), axis=1)
            res_denominator = res_denominator + res_denominator_inc
            # print "res_denominator[0]: ", res_denominator[0]
            # print "res_matrix[0]: ", res_matrix[0]
        res_matrix = res_matrix / res_denominator.reshape((-1,1)).astype(float)
            # print "res_matrix shape: ", res_matrix.shape
        ##update responsibility##
        # print "np.sum(res_matrix, axis=0): ", np.sum(res_matrix, axis=1)
        w_update, theta_vec, lambda_vec = gaussian_mixture_gd_em_descent_avg(batch_X, batch_y, res_matrix, w, theta_vec, lambda_vec, a, b, theta_alpha, C)
        # print "in avg lambda_vec_minimizer: ", lambda_vec
        w_update = alpha * w_update
        print "w_update norm: ", linalg.norm(w_update)
        w -= w_update
        print "w norm: ", linalg.norm(w)
        alpha -= alpha * decay
        k += 1

        if k % 20 == 0:
            print "non_huber_optimizator k: ", k
        if k % 60 == 0:
            print "test at step: ", k
            accuracy = testaccuracy(w, w, X_test, y_test, 'non-huber')
            print "accuracy this step: ", accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_step = k
            print "best_accuracy: ", best_accuracy
            print "best_accuracy_step: ", best_accuracy_step

        batch_iter = batch_iter + 1
        # if k >= max_iter or linalg.norm(w_update) < eps:
        # print "gaussian_mixture_gd_em_optimizator_avg k: ", k
        # if k >= max_iter:
        #     break
        if k >= max_iter or linalg.norm(w_update) < eps:
            break
        if np.isnan(linalg.norm(w)) or np.isinf(linalg.norm(w)):
            return k, w.toarray(), best_accuracy, -1
    print "gaussian_mixture_gd_em_optimizator_avg final k: ", k
    return k, w.toarray(), best_accuracy, best_accuracy_step

# theta_r for exp{theta_r}, lambda_t for exp{lambda_t}
# not checked yet!!!!!!!!!!!!!!!!
# should notice the transform from theta_r to theta, lambda_t to lambda
def gaussian_mixture_gd_optimizator_avg(X_train, y_train, X_test, y_test, C, max_iter, eps, alpha, theta_r_lr_alpha, lambda_t_lr_alpha, n_gaussian, w_init, theta_alpha, a, b, decay, batch_size, clf_name):
    k = 0
    #w = np.zeros(X_train.shape[1])
    np.random.seed(10)
    if w_init == 0.:
        w = np.zeros(X_train.shape[1])
    else:
        w = np.random.normal(0.0, w_init, X_train.shape[1])
    w = sparse.csr_matrix(w)
    print "w norm: ", linalg.norm(w)
    y_train = sparse.csr_matrix(y_train)
    y_train = y_train.T
    y_test = sparse.csr_matrix(y_test)
    y_test = y_test.T
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    print "gaussian_mixture_gd_optimizator_avg alpha: ", alpha
    print "gaussian_mixture_gd_optimizator_avg theta_r_lr_alpha: ", theta_r_lr_alpha
    print "gaussian_mixture_gd_optimizator_avg lambda_t_lr_alpha: ", lambda_t_lr_alpha
    print "gaussian_mixture_gd_optimizator_avg C: ", C
    print "gaussian_mixture_gd_optimizator_avg max_iter: ", max_iter
    print "gaussian_mixture_gd_optimizator_avg eps: ", eps
    print "gaussian_mixture_gd_optimizator_avg n_gaussian: ", n_gaussian
    print "gaussian_mixture_gd_optimizator_avg w_init: ", w_init
    print "gaussian_mixture_gd_optimizator_avg theta_alpha: ", theta_alpha
    print "gaussian_mixture_gd_optimizator_avg a: ", a
    print "gaussian_mixture_gd_optimizator_avg b: ", b
    print "gaussian_mixture_gd_optimizator_avg decay: ", decay
    print "gaussian_mixture_gd_optimizator_avg batch_size: ", batch_size
    accuracy = 0.0
    best_accuracy = 0.0
    best_accuracy_step = 0
    batch_iter = 0
    # np.random.seed(10)
    idx = np.random.permutation(X_train.shape[0])
    print "data idx: ", idx
    X_train = X_train[idx]
    y_train = y_train[idx]
    # print "in optimizator_avg b: ", b
    # initialization !!
    theta_r_vec = np.zeros(n_gaussian)
    theta_r_exp_vec = np.exp(theta_r_vec)
    theta_vec = theta_r_exp_vec / np.sum(theta_r_exp_vec)
    print "theta_vec initialization: ", theta_vec
    # lambda_t_vec = np.random.normal(0.0, 1, n_gaussian)
    lambda_t_vec = np.zeros(n_gaussian)
    for i in range(n_gaussian):
        lambda_t_vec[i] = (i+1) * np.log(1/2.)
    # lambda_t_vec = np.array([np.log(1/2.), np.log(1/4.), np.log(1/8.)])
    # lambda_t_vec = np.array([np.log(1/2.)])
    # lambda_vec = np.exp(lambda_t_vec)
    lambda_vec = np.exp(lambda_t_vec)
    while True:
        # sparse matrix works, random.shuffle
        # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
        # print "X.shape: ", X.shape
        # print "max idx: ", max(idx)
        index = (batch_size * batch_iter) % X_train.shape[0]

        if (index + batch_size) >= X_train.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

        batch_X, batch_y = X_train[index : (index + batch_size)], y_train[index : (index + batch_size)]

        ##update responsibility##
        print "theta_vec: ", theta_vec
        print "lambda_vec: ", lambda_vec
        w_array = w.toarray() # in order for np.exp
        w_array = np.reshape(w_array, w_array.shape[1])
        res_denominator = np.zeros(w_array.shape[0]-1)
        for i in range(theta_vec.shape[0]):
            # print "gaussian theta i: ", i
            res_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array[:-1] * w_array[:-1])
            if i == 0:
                res_matrix = np.reshape(res_denominator_inc, (-1, 1))
            else:
                res_matrix = np.concatenate((res_matrix, np.reshape(res_denominator_inc, (-1, 1))), axis=1)
            res_denominator = res_denominator + res_denominator_inc
        res_matrix = res_matrix / res_denominator.reshape((-1,1)).astype(float)
        # print "res_matrix shape: ", res_matrix.shape
        # print "np.sum(res_matrix, axis=1): ", np.sum(res_matrix, axis=1)
        ##update responsibility##

        w_update, theta_r_vec_update, lambda_t_vec_update = gaussian_mixture_gd_descent_avg(batch_X, batch_y, res_matrix, w, theta_r_vec, theta_vec, lambda_t_vec, lambda_vec, a, b, theta_alpha, C)
        w_update = alpha * w_update
        theta_r_vec_update = theta_r_lr_alpha * theta_r_vec_update
        lambda_t_vec_update = lambda_t_lr_alpha * lambda_t_vec_update
        print "w_update norm: ", linalg.norm(w_update)
        w -= w_update
        theta_r_vec -= theta_r_vec_update
        lambda_t_vec -= lambda_t_vec_update
        print "w norm: ", linalg.norm(w)
        #############################################
        ## update theta_vec, theta_r_vec, lambda_vec, lambda_t_vec simultaneously!!!!!!!!
        theta_r_exp_vec = np.exp(theta_r_vec)
        theta_vec = theta_r_exp_vec / np.sum(theta_r_exp_vec)
        # lambda_vec = np.exp(lambda_t_vec)
        lambda_vec = np.exp(lambda_t_vec)
        print "theta_vec: ", theta_vec
        print "lambda_vec: ", lambda_vec
        #############################################
        alpha -= alpha * decay
        theta_r_lr_alpha -= theta_r_lr_alpha * decay
        lambda_t_lr_alpha -= lambda_t_lr_alpha * decay
        k += 1
        print "non_huber_optimizator k: ", k
        if k % 20 == 0:
            print "non_huber_optimizator k: ", k
        if k % 20 == 0:
            print "test at step: ", k
            accuracy = testaccuracy(w, w, X_test, y_test, 'non-huber')
            print "accuracy this step: ", accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_step = k
            print "best_accuracy: ", best_accuracy
            print "best_accuracy_step: ", best_accuracy_step

        batch_iter = batch_iter + 1
        # if k >= max_iter or linalg.norm(w_update) < eps:
        # if k >= max_iter:
        if k >= max_iter or linalg.norm(w_update) < eps:
            break
        if np.isnan(linalg.norm(w)) or np.isinf(linalg.norm(w)):
            return k, w.toarray(), best_accuracy, -1
    print "gaussian_mixture_gd_optimizator_avg final k: ", k
    return k, w.toarray(), best_accuracy, best_accuracy_step
