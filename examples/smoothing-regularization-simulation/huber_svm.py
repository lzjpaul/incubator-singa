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
            # print "using huber_optimizator"
            # print "self.batch_size: ", self.batch_size
        #    self.n_iter_, self.w_, self.v_ = optimizator_gd.huber_optimizator(X, y, self.lambd, self.mu, self.C,
        #                                                self.max_iter, self.eps, self.alpha, self.decay, self.batch_size)
        #else:
            # print "using huber_optimizator_avg"
            # print "self.batch_size: ", self.batch_size
        self.n_iter_, self.w_, self.v_, self.best_accuracy_, self.best_accuracy_step_ = optimizator_gd.huber_optimizator_avg(X_train, y_train, X_test, y_test, self.lambd, self.mu, self.C,
                                                    (y_train.shape[0] * 20 / self.batch_size), self.eps, self.alpha, self.decay, self.batch_size, 'huber')
        self.coef_ = np.add(self.w_, self.v_).reshape((1, X_train.shape[1]))
        self.intercept_ = 0.0
        if self.fit_intercept:
            self.intercept_ = self.coef_[:,-1]
            self.coef_ = self.coef_[:,:-1]
        return self.best_accuracy_, self.best_accuracy_step_

    def predict_proba(self, X_train):
        return super(HuberSVC, self).decision_function(X_train)

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
