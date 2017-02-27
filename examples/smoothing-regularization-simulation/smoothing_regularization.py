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

class Smoothing_Regularization(BaseEstimator, LogisticLinearClassifierMixin):

    def __init__(self, C=1., lambd=1., max_iter=1000, eps=0.0001, alpha=0.01, decay=0.01, fit_intercept=True, batch_size=30):
        self.C = C
        self.lambd = lambd
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

    def fit(self, X_train, y_train, X_test, y_test):
        print "fit begin y: ", y_train[0:100]
        self.classes_, y_train = np.unique(y_train, return_inverse=True)
        self.classes_, y_test = np.unique(y_test, return_inverse=True)
        print "in Smoothing_Regularization fit"
        print "fit X shape: ", X_train.shape
        # print "fit X norm: ", np.linalg.norm(X)
        print "fit y shape: ", y_train.shape
        print "fit y norm: ", np.linalg.norm(y_train)
        print "fit y after unique: ", y_train[0:100]
        print "fit y after unique self.classes_: ", self.classes_
        if min(y_train) == 0:
            y_train = 2. * y_train - 1
            y_test = 2. * y_test - 1
        print "y_train[0:100]: ", y_train[0:100]
        print "y_test[0:100]: ", y_test[0:100]
        if self.fit_intercept:
            if sparse.issparse(X_train):
                # X = np.hstack((X.toarray(), np.ones((X.shape[0], 1))))
                # X = csr_matrix(X)
                # X = sparse.hstack((X, csr_matrix(np.ones((X.shape[0], 1))))).tocsr()
                X_train = sparse.hstack([X_train, np.ones((X_train.shape[0], 1))], format="csr")
                X_test = sparse.hstack([X_test, np.ones((X_test.shape[0], 1))], format="csr")
            else:
                X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
                X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
        #if self.gradaverage == 0:
            # print "using smoothing_optimizator"
            # print "self.batch_size: ", self.batch_size
        #    self.n_iter_, self.w_ = optimizator_gd.smoothing_optimizator(X, y, self.lambd, self.C,
        #                                                self.max_iter, self.eps, self.alpha, self.decay, self.batch_size)
        #else:
            # print "using smoothing_optimizator_avg"
        print "self.lambd: ", self.lambd
        print "self.alpha: ", self.alpha
        print "self.C: ", self.C
        self.n_iter_, self.w_, self.best_accuracy_, self.best_accuracy_step_ = optimizator_gd.non_huber_optimizator_avg(X_train, y_train, X_test, y_test, self.lambd, 0., self.C,
                                                     (y_train.shape[0] * 20 / self.batch_size), self.eps, self.alpha, self.decay, self.batch_size, 'smoothing')
        self.coef_ = self.w_.reshape((1, X_train.shape[1]))
        self.intercept_ = 0.0
        if self.fit_intercept:
            self.intercept_ = self.coef_[:,-1]
            self.coef_ = self.coef_[:,:-1]
        print "in Smoothing_Regularization fit"
        print "fit self.intercept_ shape: ", self.intercept_.shape
        print "fit self.intercept_ norm: ", np.linalg.norm(self.intercept_)
        print "fit self.coef_ shape: ", self.coef_.shape
        return self.best_accuracy_, self.best_accuracy_step_

    def predict_proba(self, X_train):
        return super(Smoothing_Regularization, self).decision_function(X_train)

    def get_params(self, deep=True):
        return {'lambd': self.lambd,
                'C': self.C,
                'max_iter': self.max_iter,
                'eps': self.eps,
                'alpha': self.alpha,
                'decay': self.decay,
                'fit_intercept': self.fit_intercept,
                # 'gradaverage': self.gradaverage,
                'batch_size': self.batch_size}
