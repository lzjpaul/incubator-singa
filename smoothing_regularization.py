__author__ = "Luo Zhaojing"
__version__ = "0.1.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator

def smoothing_grad_descent(X, y, w, param, C):
    grad =  (np.exp(w) - 1) / (np.exp(w) + 1) # log(1+e^(-w)) + log(1+e^(w))
    f1 = np.exp(-y * np.dot(w, X.T))
    res = np.repeat((C * -y * (f1 / (1.0 + f1))).reshape(X.shape[0], 1), X.shape[1], axis=1) * X
    return grad + res.sum(axis=0)

def smoothing_optimizator(X, y, lambd, C, max_iter, eps, alpha, decay):
    k = 0
    w = np.zeros(X.shape[1])
    
    while True:
        vec = w
        
        # making optimization in w
        w -= alpha * smoothing_grad_descent(X, y, w, lambd, C)
        
        alpha -= alpha * decay
        k += 1

        if k >= max_iter or np.linalg.norm(w - vec, ord=2) < eps:
            break

    return k, w

class Smoothing_Regularization(BaseEstimator, LinearClassifierMixin):

    def __init__(self, lambd=1., C=10., max_iter=1000, eps=0.0001, alpha=0.01, decay=0.01, fit_intercept=True):
        self.lambd = lambd
        self.C = C
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        y = 2. * y - 1
        if self.fit_intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.n_iter_, self.w_, = smoothing_optimizator(X, y, self.lambd, self.C, 
                                                     self.max_iter, self.eps, self.alpha, self.decay)
        self.coef_ = self.w_.reshape((1, X.shape[1]))
        self.intercept_ = 0.0
        if self.fit_intercept:
            self.intercept_ = self.coef_[:,-1]
            self.coef_ = self.coef_[:,:-1]
        return self
    
    def predict_proba(self, X):
        return super(Smoothing_Regularization, self)._predict_proba_lr(X)
    
    def get_params(self, deep=True):
        return {'lambd': self.lambd,
                'C': self.C, 
                'max_iter': self.max_iter, 
                'eps': self.eps, 
                'alpha': self.alpha,
                'decay': self.decay,
                'fit_intercept': self.fit_intercept}
