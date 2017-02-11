# _3: bias not sampled
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
    grad[:-1] = grad_numerator / grad_denominator # -log(p(w))
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
    print "param * 2.0: ", param * 2.0
    grad = param * 2.0 * w
    grad = grad.toarray()
    # print "before grad: ", grad[0,-2]
    grad[0, -1] = 0.0
    print "after grad: ", grad[0, 0:10]
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
        if k >= max_iter:
            break
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
        if k >= max_iter:
            break
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
            ############LDA_sampler#################
            print "before sampler w: ", linalg.norm(w)
            theta_vec, lambda_vec = sampler.run(np.reshape((w.toarray()*np.sqrt(9375))[0,:-1], (w.toarray().shape[1]-1)), (k-initial_L2_step), batchgibbs)
            print "theta_vec: ", theta_vec
            print "lambda_vec: ", lambda_vec
            # lambda_vec[0] = 0.2
            # print "lambda_vec enforce: ", lambda_vec
            ############LDA_sampler#################
            w_update = alpha * gaussian_mixture_descent_avg(batch_X, batch_y, w, theta_vec, lambda_vec, C)
            print "w_update norm: ", linalg.norm(w_update)
            w -= w_update
            print "w: ", linalg.norm(w)
            # print "w: ", w
            alpha -= alpha * decay
            print "lr alpha: ", alpha
        else: #L2_norm
            # w_update = initial_L2_alpha * ridge_grad_descent_avg(batch_X, batch_y, w, initial_L2_lambd, 0.1, C)
            w_update = alpha * ridge_grad_descent_avg(batch_X, batch_y, w, initial_L2_lambd, 0.1, C)
            print "w_update norm: ", linalg.norm(w_update)
            w -= w_update
            print "w: ", linalg.norm(w)
            # print "w: ", w
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
        if k >= max_iter:
        # if k >= 20:
            break
    print "gaussian opt avg final k: ", k
    return k, w.toarray(), best_accuracy, best_accuracy_step
