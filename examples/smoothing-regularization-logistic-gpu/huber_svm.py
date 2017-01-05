__author__ = "Gunthard Benecke and Oleksandr Zadorozhnyi"
__license__ = "GPL"
__version__ = "0.2.0"

import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from lr_linear_mixin import LogisticLinearClassifierMixin
from scipy import sparse
import optimizator_gd

class HuberSVC(BaseEstimator, LogisticLinearClassifierMixin):

    def __init__(self, C=1., lambd=1., mu=1., max_iter=1000, eps=0.0001, alpha=0.01, decay=0.01, fit_intercept=True, batch_size=30):
        self.lambd = lambd
        self.mu = mu
        self.C = C
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size
        # self.gradaverage = gradaverage

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        y = 2. * y - 1
        #if self.fit_intercept:
        #    X = np.hstack((X, np.ones((X.shape[0], 1))))
        if self.fit_intercept:
            if sparse.issparse(X):
                # X = np.hstack((X.toarray(), np.ones((X.shape[0], 1))))
                # X = csr_matrix(X)
                # X = sparse.hstack((X, csr_matrix(np.ones((X.shape[0], 1))))).tocsr()
                X = sparse.hstack([X, np.ones((X.shape[0], 1))], format="csr")
            else:
                X = np.hstack((X, np.ones((X.shape[0], 1))))
        #if self.gradaverage == 0:
            # print "using huber_optimizator"
            # print "self.batch_size: ", self.batch_size
        #    self.n_iter_, self.w_, self.v_ = optimizator_gd.huber_optimizator(X, y, self.lambd, self.mu, self.C,
        #                                                self.max_iter, self.eps, self.alpha, self.decay, self.batch_size)
        #else:
            # print "using huber_optimizator_avg"
            # print "self.batch_size: ", self.batch_size
        self.n_iter_, self.w_, self.v_ = optimizator_gd.huber_optimizator_avg(X, y, self.lambd, self.mu, self.C,
                                                    self.max_iter, self.eps, self.alpha, self.decay, self.batch_size, 'huber')
        self.coef_ = np.add(self.w_, self.v_).reshape((1, X.shape[1]))
        self.intercept_ = 0.0
        if self.fit_intercept:
            self.intercept_ = self.coef_[:,-1]
            self.coef_ = self.coef_[:,:-1]
        return self

    def predict_proba(self, X):
        return super(HuberSVC, self).decision_function(X)

    def get_params(self, deep=True):
        return {'lambd': self.lambd,
                'mu': self.mu,
                'C': self.C,
                'max_iter': self.max_iter,
                'eps': self.eps,
                'alpha': self.alpha,
                'decay': self.decay,
                'fit_intercept': self.fit_intercept,
                # 'gradaverage': self.gradaverage,
                'batch_size': self.batch_size}
