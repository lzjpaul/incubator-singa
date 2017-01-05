# ?? batchsize * batch_iter / shape[0]?
# gradient descent: divided by # of samples --> float?
__author__ = "Luo Zhaojing"
__version__ = "0.1.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
import random
from scipy import sparse
from scipy.sparse import csr_matrix
import os
from singa import initializer
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor


def huber_grad_descent_avg(batch_X, batch_y, w, v, param, C, is_l1):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in huber gd avg"
    grad = param * np.sign(v) if is_l1 else param * 2.0 * w
    f1 = np.exp(-batch_y * np.dot(w + v, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    ressum = res.sum(axis=0)
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + ressum

def lasso_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C, dev):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in lasso gd avg"
    grad = param * np.sign(w)
    f1 = np.exp(-batch_y * np.dot(w, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    ressum = res.sum(axis=0)
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + ressum

def ridge_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C, dev):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in ridge gd avg"
    grad = param * 2.0 * w
    f1 = np.exp(-batch_y * np.dot(w, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    ressum = res.sum(axis=0)
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + ressum

def elasticnet_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C, dev):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in elastic gd avg"
    grad = param * l1_ratio_or_mu * np.sign(w) + param * (1 - l1_ratio_or_mu) * w
    f1 = np.exp(-batch_y * np.dot(w, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    ressum = res.sum(axis=0)
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + ressum


def smoothing_grad_descent_avg(batch_X, batch_y, w, param, l1_ratio_or_mu, C, dev):
    # print "begin toarray"
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "end toarray"
    # print "in smoothing gd avg"

    tbatch_X = tensor.from_numpy(batch_X)
    tbatch_X.to_device(dev)
    w = w.reshape(1, w.shape[0])
    tw = tensor.from_numpy(w)
    tw.to_device(dev)
    tbatch_y = tensor.from_numpy(batch_y)
    tbatch_y.to_device(dev)
    
    # grad =  param * (np.exp(w) - 1) / (np.exp(w) + 1) # log(1+e^(-w)) + log(1+e^(w))
    tgrad =  param * ((tensor.exp(tw) - 1.) / (tensor.exp(tw) + 1.)) # log(1+e^(-w)) + log(1+e^(w))
    # print "grad: ", grad
    # print "float(batch_X.shape[0]): ", float(batch_X.shape[0])
    # print 'grad.shape: ', grad.shape
    # f1 = np.exp(-batch_y * np.dot(w, batch_X.T))
    tf1 = tensor.exp( ((-1) * tbatch_y) * tensor.mult(tw, tbatch_X.T()) )
    # res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    # print 'res.shape: ', res.shape
    # print 'res.sum(axis=0) shape: ', res.sum(axis=0).shape
    # ressum = res.sum(axis=0)
    # ressum = ressum.astype(np.float)
    tres = tbatch_X
    tres.mult_column(C * ((-1) * tbatch_y) * (tf1 / (1.0 + tf1)))
    tressum = tensor.sum(tres, 0)
    # ressum /=  float(batch_X.shape[0])
    tressum = tressum / float(batch_X.shape[0])
    # print "end smoothing gd"
    return tgrad + tressum

def huber_optimizator_avg(X, y, lambd, l1_ratio_or_mu, C, max_iter, eps, alpha, decay, batch_size, clf_name):
    k = 0
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])

    batch_iter = 0
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    grad_descent_avg = huber_grad_descent_avg
    while True:
        index = (batch_size * batch_iter) % X.shape[0]

        if (index + batch_size) >= X.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

        batch_X, batch_y = X[index : (index + batch_size)], y[index : (index + batch_size)]

        vec = np.add(w, v)
        # making optimization in w and v
        v -= alpha * grad_descent_avg(batch_X, batch_y, w, v, l1_ratio_or_mu, C, True)
        w -= alpha * grad_descent_avg(batch_X, batch_y, w, v, lambd, C, False)

        alpha -= alpha * decay
        k += 1
        print "huber_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(np.add(w,v) - vec, ord=2) < eps:
            break
    print "huber opt avg final k: ", k
    return k, w, v

# only one weight w, not w + v
def non_huber_optimizator_avg(X, y, lambd, l1_ratio_or_mu, C, max_iter, eps, alpha, decay, batch_size, clf_name):
    k = 0
    w = np.zeros(X.shape[1])
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    
    dev = device.create_cuda_gpu()
    cpudev = device.get_default_device()
 
    batch_iter = 0
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
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
        index = (batch_size * batch_iter) % X.shape[0]

        if (index + batch_size) >= X.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

        batch_X, batch_y = X[index : (index + batch_size)], y[index : (index + batch_size)]
        tgd_avg = grad_descent_avg(batch_X, batch_y, w, lambd, l1_ratio_or_mu, C, dev)
        # print "return in tgd_avg"
        tgd_avg.to_device(cpudev)
        # print "finish tgd_avg.to_device(cpudev)"
        # w_update = alpha * grad_descent_avg(batch_X, batch_y, w, lambd, l1_ratio_or_mu, C, dev)
        gd_avg = tensor.to_numpy(tgd_avg)
        w_update = alpha * gd_avg
        w_update = np.reshape(w_update, w_update.shape[1])
        # print "w_update shape: ", w_update.shape
        # print "w shape: ", w.shape
        w -= w_update
        alpha -= alpha * decay
        k += 1
        if k % 2 == 0:
            print "smoothing_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(w_update, ord=2) < eps:
            break
    print "smoothing opt avg final k: ", k
    return k, w
