# 2-27: split into train and valid, test and valid accuracy
# size of all the training samples should be passed to the gradient calculation also
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

def gaussian_mixture_gd_descent_avg(trainvalid_num, batch_X, batch_y, res_matrix, w, theta_r_vec, theta_vec, lambda_t_vec, lambda_vec, a, b, theta_alpha, C):
    print "trainvalid_num: ", trainvalid_num
    # data preprocess
    batch_y = batch_y.T
    w_array = w.toarray() # in order for np.exp
    w_array = np.reshape(w_array, w_array.shape[1])
    w_weight_array = w_array[:-1].reshape((-1, 1)) # no bias

    # calculate gaussian mixture regularization
    lambda_w = np.dot(w_weight_array, lambda_vec.reshape((1, -1)))
    grad = np.zeros(w_array.shape[0])
    grad[:-1] = np.sum((res_matrix * lambda_w), axis=1)# -log(p(w))
    grad[-1] = 0.0

    # calculate data gradient
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    ressum *= trainvalid_num
    print "ressum *= trainvalid_num"
    ###lambda_t_update#########
    print "float(a-1): ", float(a-1)
    term1 = (float(a-1) / lambda_vec) - b
    print "term1: ", term1
    res_w = sparse.csr_matrix(res_matrix.T).dot(sparse.diags(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])))
    res_w = (res_w.toarray().T)
    term2 = np.sum((res_matrix / (2.0 * lambda_vec.reshape(1, -1))) - res_w, axis=0)
    lambda_t_vec_update = (- term1 - term2) * lambda_vec # lambda = exp(lambda_t)derivative of lambda to t
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


    ###theta_r_update#########
    return sparse.csr_matrix(grad + ressum), theta_r_vec_update, lambda_t_vec_update

def ridge_grad_descent_avg(trainvalid_num, batch_X, batch_y, w, param, l1_ratio_or_mu, C):
    print "trainvalid_num: ", trainvalid_num
    # data preprocess
    batch_y = batch_y.T
    # ridge regularization
    grad = param * 2.0 * w
    grad = grad.toarray()
    grad[0, -1] = 0.0 # bias
    grad = sparse.csr_matrix(grad)
    # data gradient
    f1 = np.exp(((-batch_y).multiply(w.dot(batch_X.T))).toarray())
    y_res = (C * ((-batch_y).toarray()*(f1 / (1.0 + f1))))
    y_res = np.asarray(y_res).reshape(f1.shape[1])
    res = batch_X.T.dot(sparse.csr_matrix(sparse.diags(y_res, 0)))
    res = res.T
    ressum = res.sum(axis=0) #this will turn back to dense
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    ressum *= trainvalid_num
    print "ressum *= trainvalid_num"
    return grad + sparse.csr_matrix(ressum)


# only one weight w, not w + v
def non_huber_optimizator_avg(X_train, y_train, X_test, y_test, lambd, l1_ratio_or_mu, C, max_iter, eps, alpha, decay, batch_size, clf_name):
    k = 0
    # w = np.zeros(X_train.shape[1])
    w = np.random.normal(0.0, 0.0001, X_train.shape[1])
    w = sparse.csr_matrix(w)
    ### split train and valid ###
    validation_perc = 0.3
    validationNum = int(validation_perc*X_train.shape[0])
    print "y_train shape: ", y_train.shape
    X_valid, y_valid = X_train[:validationNum, ], y_train[:validationNum]
    X_train, y_train = X_train[validationNum:, ], y_train[validationNum:]
    ### split train and valid ###
    y_train = sparse.csr_matrix(y_train)
    y_train = y_train.T
    y_test = sparse.csr_matrix(y_test)
    y_test = y_test.T
    y_valid = sparse.csr_matrix(y_valid)
    y_valid = y_valid.T
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    X_valid = sparse.csr_matrix(X_valid)
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    print "optimizator_avg lambd: ", lambd
    print "optimizator_avg alpha: ", alpha
    print "optimizator_avg max_iter: ", max_iter
    best_valid_accuracy = 0.0
    best_valid_accuracy_step = 0
    best_test_accuracy = 0.0
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
        index = (batch_size * batch_iter) % X_train.shape[0]

        if (index + batch_size) >= X_train.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            np.random.seed(k)
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

        batch_X, batch_y = X_train[index : (index + batch_size)], y_train[index : (index + batch_size)]
        w_update = alpha * grad_descent_avg((X_train.shape[0] + X_valid.shape[0]), batch_X, batch_y, w, lambd, l1_ratio_or_mu, C)
        w -= w_update
        alpha -= alpha * decay
        k += 1

        if k % 60 == 0:
            print "valid at step: ", k
            valid_accuracy = testaccuracy(w, w, X_valid, y_valid, 'non-huber')
            test_accuracy = testaccuracy(w, w, X_test, y_test, 'non-huber')
            print "valid_accuracy this step: ", valid_accuracy
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_valid_accuracy_step = k
                best_test_accuracy = test_accuracy
            print "best_valid_accuracy: ", best_valid_accuracy
            print "best_valid_accuracy_step: ", best_valid_accuracy_step
            print "best_test_accuracy: ", best_test_accuracy

        batch_iter = batch_iter + 1
        if k >= max_iter or linalg.norm(w_update/float(alpha)) < eps:
            break
        if np.isnan(linalg.norm(w)) or np.isinf(linalg.norm(w)):
            return k, w.toarray(), best_test_accuracy, -1
    print "non_huber opt avg final k: ", k
    return k, w.toarray(), best_test_accuracy, best_valid_accuracy_step

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
    ### split train and valid ###
    validation_perc = 0.3
    validationNum = int(validation_perc*X_train.shape[0])
    X_valid, y_valid = X_train[:validationNum, ], y_train[:validationNum]
    X_train, y_train = X_train[validationNum:, ], y_train[validationNum:]
    ### split train and valid ###
    y_train = sparse.csr_matrix(y_train)
    y_train = y_train.T
    y_test = sparse.csr_matrix(y_test)
    y_test = y_test.T
    y_valid = sparse.csr_matrix(y_valid)
    y_valid = y_valid.T
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    X_valid = sparse.csr_matrix(X_valid)
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
    best_valid_accuracy = 0.0
    best_valid_accuracy_step = 0
    best_test_accuracy = 0.0
    batch_iter = 0
    # np.random.seed(10)
    idx = np.random.permutation(X_train.shape[0])
    print "data idx: ", idx
    X_train = X_train[idx]
    y_train = y_train[idx]
    # initialization !!
    theta_r_vec = np.zeros(n_gaussian)
    theta_r_exp_vec = np.exp(theta_r_vec)
    theta_vec = theta_r_exp_vec / np.sum(theta_r_exp_vec)
    print "theta_vec initialization: ", theta_vec
    lambda_t_vec = np.zeros(n_gaussian)
    for i in range(n_gaussian):
        lambda_t_vec[i] = (i+1) * np.log(1/2.)
    lambda_vec = np.exp(lambda_t_vec)
    while True:
        # sparse matrix works, random.shuffle
        # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
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
            res_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array[:-1] * w_array[:-1])
            if i == 0:
                res_matrix = np.reshape(res_denominator_inc, (-1, 1))
            else:
                res_matrix = np.concatenate((res_matrix, np.reshape(res_denominator_inc, (-1, 1))), axis=1)
            res_denominator = res_denominator + res_denominator_inc
        res_matrix = res_matrix / res_denominator.reshape((-1,1)).astype(float)
        ##update responsibility##

        w_update, theta_r_vec_update, lambda_t_vec_update = gaussian_mixture_gd_descent_avg((X_train.shape[0] + X_valid.shape[0]), batch_X, batch_y, res_matrix, w, theta_r_vec, theta_vec, lambda_t_vec, lambda_vec, a, b, theta_alpha, C)
        w_update = alpha * w_update
        theta_r_vec_update = theta_r_lr_alpha * theta_r_vec_update
        lambda_t_vec_update = lambda_t_lr_alpha * lambda_t_vec_update
        w -= w_update
        theta_r_vec -= theta_r_vec_update
        lambda_t_vec -= lambda_t_vec_update
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
        if k % 60 == 0:
            print "valid at step: ", k
            valid_accuracy = testaccuracy(w, w, X_valid, y_valid, 'non-huber')
            test_accuracy = testaccuracy(w, w, X_test, y_test, 'non-huber')
            print "valid_accuracy this step: ", valid_accuracy
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_valid_accuracy_step = k
                best_test_accuracy = test_accuracy
            print "best_valid_accuracy: ", best_valid_accuracy
            print "best_valid_accuracy_step: ", best_valid_accuracy_step
            print "best_test_accuracy: ", best_test_accuracy
        batch_iter = batch_iter + 1
        if k >= max_iter or linalg.norm(w_update/float(alpha)) < eps:
            break
        if np.isnan(linalg.norm(w)) or np.isinf(linalg.norm(w)):
            return k, w.toarray(), best_test_accuracy, -1
    print "gaussian_mixture_gd_optimizator_avg final k: ", k
    return k, w.toarray(), best_test_accuracy, best_valid_accuracy_step
