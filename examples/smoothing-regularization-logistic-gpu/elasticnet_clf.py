#11-23 adding parama lambda for tuning parameters
#11-29 move x to optimization
__author__ = "Luo Zhaojing"
__version__ = "0.1.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from lr_linear_mixin import LogisticLinearClassifierMixin
import random
from scipy import sparse
from scipy.sparse import csr_matrix
import optimizator_gd

class Elasticnet_Classifier(BaseEstimator, LogisticLinearClassifierMixin):

    def __init__(self, C=1., lambd=1., l1_ratio = 0.01, max_iter=1000, eps=0.0001, alpha=0.01, decay=0.01, fit_intercept=True, batch_size=30):
        self.C = C
        self.lambd = lambd
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size
        # self.gradaverage = gradaverage
        # print "init self.C: ", self.C
        # print "init self.lambd: ", self.lambd
        # print "init self.batch_size: ", self.batch_size
        # print "init self.gradaverage: ", self.gradaverage

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        y = 2. * y - 1
        if self.fit_intercept:
            if sparse.issparse(X):
                # X = np.hstack((X.toarray(), np.ones((X.shape[0], 1))))
                # X = csr_matrix(X)
                # X = sparse.hstack((X, csr_matrix(np.ones((X.shape[0], 1))))).tocsr()
                X = sparse.hstack([X, np.ones((X.shape[0], 1))], format="csr")
            else:
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        #if self.gradaverage == 0:
            # print "using smoothing_optimizator"
            # print "self.batch_size: ", self.batch_size
        #    self.n_iter_, self.w_ = optimizator_gd.smoothing_optimizator(X, y, self.lambd, self.C,
        #                                                self.max_iter, self.eps, self.alpha, self.decay, self.batch_size)
        #else:
            # print "using smoothing_optimizator_avg"
            # print "self.batch_size: ", self.batch_size
        self.n_iter_, self.w_ = optimizator_gd.non_huber_optimizator_avg(X, y, self.lambd, self.l1_ratio, self.C,
                                                    self.max_iter, self.eps, self.alpha, self.decay, self.batch_size, 'elasticnet')
        self.coef_ = self.w_.reshape((1, X.shape[1]))
        self.intercept_ = 0.0
        if self.fit_intercept:
            self.intercept_ = self.coef_[:,-1]
            self.coef_ = self.coef_[:,:-1]
        return self

    def predict_proba(self, X):
        return super(Elasticnet_Classfier, self).decision_function(X)

    def get_params(self, deep=True):
        return {'lambd': self.lambd,
                'C': self.C,
                'l1_ratio': self.l1_ratio,
                'max_iter': self.max_iter,
                'eps': self.eps,
                'alpha': self.alpha,
                'decay': self.decay,
                'fit_intercept': self.fit_intercept,
                # 'gradaverage': self.gradaverage,
                'batch_size': self.batch_size}
