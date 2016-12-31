# Singa: stopping criteria?
# initialization?
# avg?
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

def huber_grad_descent(batch_X, batch_y, w, v, param, C, is_l1):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()

    grad = param * np.sign(v) if is_l1 else param * 2.0 * w
    f1 = np.exp(-batch_y * np.dot(w + v, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    return grad + res.sum(axis=0)

def huber_grad_descent_avg(batch_X, batch_y, w, v, param, C, is_l1):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()

    grad = param * np.sign(v) if is_l1 else param * 2.0 * w
    f1 = np.exp(-batch_y * np.dot(w + v, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    ressum = res.sum(axis=0)
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + ressum


def smoothing_grad_descent(batch_X, batch_y, w, param, C):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()

    grad =  param * (np.exp(w) - 1) / (np.exp(w) + 1) # log(1+e^(-w)) + log(1+e^(w))
    # print 'grad.shape: ', grad.shape
    f1 = np.exp(-batch_y * np.dot(w, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    # print 'res.shape: ', res.shape
    # print 'res.sum(axis=0) shape: ', res.sum(axis=0).shape
    return grad + res.sum(axis=0)

def smoothing_grad_descent_gpu(batch_X, batch_y, w, param, C):
    dev = device.create_cuda_gpu()

    if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    
    tbatch_X = tensor.from_numpy(batch_X)
    tbatch_X.to_device(dev)
    tw = tensor.from_numpy(w)
    tw.to_device(dev)
    tbatch_y = tensor.from_numpy(batch_y)
    tbatch_y.to_device(dev)

    tgrad =  param * ((tensor.exp(tw) - 1.) / (tensor.exp(w) + 1.)) # log(1+e^(-w)) + log(1+e^(w))
    # print 'grad.shape: ', grad.shape
    tf1 = tensor.exp( ((-1) * tbatch_y) * tensor.mult(tw, tbatch_X.T()) )
    tres = tbatch_X
    tres.mult_column(C * ((-1) * tbatch_y) * (tf1 / (1.0 + tf1)))
    # print 'res.shape: ', res.shape
    # print 'res.sum(axis=0) shape: ', res.sum(axis=0).shape
    # !!!! here regularization no need to divided by /batchsize!!
    return tgrad + tensor.sum(tres, 0) # here regularization no need to divided by /batchsize!!
    # !!!! here regularization no need to divided by /batchsize!!

def smoothing_grad_descent_avg(batch_X, batch_y, w, param, C):
    if sparse.issparse(batch_X): batch_X = batch_X.toarray()

    grad =  param * (np.exp(w) - 1) / (np.exp(w) + 1) # log(1+e^(-w)) + log(1+e^(w))
    # print 'grad.shape: ', grad.shape
    f1 = np.exp(-batch_y * np.dot(w, batch_X.T))
    res = np.repeat((C * -batch_y * (f1 / (1.0 + f1))).reshape(batch_X.shape[0], 1), batch_X.shape[1], axis=1) * batch_X
    # print 'res.shape: ', res.shape
    # print 'res.sum(axis=0) shape: ', res.sum(axis=0).shape
    ressum = res.sum(axis=0)
    ressum = ressum.astype(np.float)
    ressum /=  float(batch_X.shape[0])
    return grad + ressum



def huber_optimizator(X, y, lambd, mu, C, max_iter, eps, alpha, decay, batch_size):
    k = 0
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])
    
    batch_iter = 0
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
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
        v -= alpha * huber_grad_descent(batch_X, batch_y, w, v, mu, C, True)
        w -= alpha * huber_grad_descent(batch_X, batch_y, w, v, lambd, C, False)
        
        alpha -= alpha * decay
        k += 1

        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(np.add(w,v) - vec, ord=2) < eps:
            break
    print "huber opt final k: ", k
    return k, w, v

def huber_optimizator_avg(X, y, lambd, mu, C, max_iter, eps, alpha, decay, batch_size):
    k = 0
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])
    
    batch_iter = 0
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
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
        v -= alpha * huber_grad_descent_avg(batch_X, batch_y, w, v, mu, C, True)
        w -= alpha * huber_grad_descent_avg(batch_X, batch_y, w, v, lambd, C, False)
        
        alpha -= alpha * decay
        k += 1

        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(np.add(w,v) - vec, ord=2) < eps:
            break
    print "huber opt avg final k: ", k
    return k, w, v


def smoothing_optimizator_singa(X, y, lambd, C, max_iter, eps, alpha, decay, batch_size, use_gpu=True):
    k = 0
    # w = np.zeros(X.shape[1])
    tw = tensor.from_numpy(np.zeros(X.shape[1], dtype = np.float32)) 
    # tw = tensor.Tensor((X.shape[1]))
    tw.gaussian(0.0, 0.1)
    lr = alpha
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    opt = optimizer.SGD(momentum=0.9, weight_decay=decay)

    if use_gpu:
        dev = device.create_cuda_gpu()
    else:
        dev = device.get_default_device()

    tw.to_device(dev)


    
    batch_iter = 0
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    epoch = 0
    while True:
        # sparse matrix works, random.shuffle
        # shuffle: next time shuffle index will be forgetten (static variable: smoothing_grad_descent.idx)
        # print "X.shape: ", X.shape
        # print "max idx: ", max(idx)
        index = (batch_size * batch_iter) % X.shape[0]
        
        if (index + batch_size) >= X.shape[0]: #new epoch
            index = 0
            batch_iter = 0 #new epoch
            epoch = epoch + 1
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

        batch_X, batch_y = X[index : (index + batch_size)], y[index : (index + batch_size)]
        tw.to_device(device.get_default_device())
        tgw = smoothing_grad_descent_gpu(batch_X, batch_y, tensor.to_numpy(tw), lambd, C)
        tw_update = alpha * smoothing_grad_descent_gpu(batch_X, batch_y, tensor.to_numpy(tw), lambd, C)
        tw.to_device(dev)
        tw_update.to_device(device.get_default_device())
        # w -= w_update
        alpha -= alpha * decay
        # !!!! here regularization no need to divided by /batchsize!!
        opt.apply_with_lr(epoch, lr / batch_size, tgw, tw, 'w')
        # !!!! here regularization no need to divided by /batchsize!!
        k += 1
        # if k % 200 == 0:
        #     print "smoothing_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(tensor.to_numpy(tw_update), ord=2) < eps:
            break
    print "smoothing opt final k: ", k
    return k, tensor.to_numpy(tw)

def smoothing_optimizator(X, y, lambd, C, max_iter, eps, alpha, decay, batch_size, use_gpu=False):
    k = 0
    w = np.zeros(X.shape[1])
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    
    batch_iter = 0
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
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
        w_update = alpha * smoothing_grad_descent(batch_X, batch_y, w, lambd, C)
        w -= w_update
        alpha -= alpha * decay
        k += 1
        # if k % 200 == 0:
        #     print "smoothing_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(w_update, ord=2) < eps:
            break
    print "smoothing opt final k: ", k
    return k, w

def smoothing_optimizator_avg(X, y, lambd, C, max_iter, eps, alpha, decay, batch_size, use_gpu=False):
    k = 0
    w = np.zeros(X.shape[1])
    # print "w shape: ", w.shape
    # f1 = open('outputfile', 'w+')
    
    batch_iter = 0
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
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
        w_update = alpha * smoothing_grad_descent_avg(batch_X, batch_y, w, lambd, C)
        w -= w_update
        alpha -= alpha * decay
        k += 1
        # if k % 200 == 0:
        #     print "smoothing_optimizator k: ", k
        batch_iter = batch_iter + 1
        if k >= max_iter or np.linalg.norm(w_update, ord=2) < eps:
            break
    print "smoothing opt avg final k: ", k
    return k, w
