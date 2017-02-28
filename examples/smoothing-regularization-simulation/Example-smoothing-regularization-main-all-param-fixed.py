# 2-23: n_gaussian number
# 2-18: adding gaussian-mixture-gd

# 1-15:to line 203 elastic
# logic bug: set regularization to 0 and see the scale for parameters
# python Example-iris-smoothing.py /data1/zhaojing/regularization/uci-dataset/car_evaluation/car.categorical.data 1 1
# the first 1 is label column, the second 1 is scale or not

########################important parameters##################################
# n_job = (-)1
# batchsize = 30
# y = +-1?
# sparse or not?
# batch SGD or all-data in?
# 12-15:
# ...n_job = -1
# ...batchsize
# ...gradient_average
##############################################################################
from huber_svm import HuberSVC
from lasso_clf import Lasso_Classifier
from ridge_clf import Ridge_Classifier
from elasticnet_clf import Elasticnet_Classifier
from smoothing_regularization import Smoothing_Regularization
from gaussian_mixture_regularization import Gaussian_Mixture_Regularization
from gaussian_mixture_gd_regularization import Gaussian_Mixture_GD_Regularization

import pandas
import numpy as np
from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_iris
# from sklearn.multiclass import OneVsRestClassifier
from logistic_ovr import LogisticOneVsRestClassifier
# from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
# from sklearn.preprocessing import scale
from sklearn import preprocessing
from DataLoader import classificationDataLoader
from svmlightDataLoader import svmlightclassificationDataLoader
from scipy import sparse
import warnings
import sys
import datetime
import time
import argparse
import random
# simulation data sets
import pickle
from gm_prior_simulation import Simulator

warnings.filterwarnings("ignore")

# data = load_iris()

# # X = data['data']
# X = scale(data['data'])
# y = data['target']
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-clf', type=str, help='classifier name')
    parser.add_argument('-datapath', type=str, help='simulation data')
    parser.add_argument('-batchsize', type=int, help='batchsize')
    parser.add_argument('-scale', type=int, help='scale or not')

    args = parser.parse_args()

    file = open(args.datapath,'r')
    simulator = pickle.load(file)
    X, y = simulator.x, (1 * simulator.label)
    print "X shape:", X.shape
    print "y shape: ", y.shape
    print "X[0, :10]: ", X[0, :10]
    print "y[:10]: ", y[:10]
    # debug: using scale
    if args.scale == 1:
        if sparse.issparse(X):
            dense_X = preprocessing.scale(X.toarray())
            X = sparse.csr_matrix(dense_X)
        else:
            X = preprocessing.scale(X)
        print "using scale"

    print "X.shape = \n", X.shape
    print "X dtype = \n", X.dtype
    print "y.shape = \n", y.shape
    # print "args.batchsize = ", args.batchsize

    print "isinstance(X, list): ", isinstance(X, list)
    np.random.seed(10)
    idx = np.random.permutation(X.shape[0])
    print "idx: ", idx
    X = X[idx]
    y = y[idx]

    print "classifier: ", args.clf


    param_gaussianmixturegd = {'estimator__C': [1.],
                       'estimator__batch_size': [args.batchsize],
                       'estimator__alpha': [1e-4, 1e-6, 1e-8],
                       'estimator__theta_r_lr_alpha': [1e-4, 1e-5, 1e-6], # the lr of theta_r is smaller
                       'estimator__lambda_t_lr_alpha': [1e-4, 1e-5, 1e-6], # the lr of theta_r is smaller
                       # 'estimator__alpha': [1],
                       # 'estimator__n_gaussian': [4],
                       'estimator__n_gaussian': [4],
                       'estimator__w_init': [0.0001], # variance of w initialization
                       'estimator__theta_alpha': [10, 100, 1000],
                       'estimator__a': [1, 10, 100],
                       'estimator__b': [1, 10, 100]
                      }

    # ridge = LogisticOneVsRestClassifier(Ridge_Classifier(batch_size=args.batchsize))
    param_ridge = {'estimator__C': [1.],
                   'estimator__lambd': [10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 1e-1, 1e-2],
                   'estimator__batch_size': [args.batchsize],
                   'estimator__alpha': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
                   }

    n_folds = 5
    # param_folds = 3
    # scoring = 'accuracy'

    result_df = pandas.DataFrame()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        print "i: ", i
        print "train_index: ", train_index
        print "test_index: ", test_index
        if i > 0:
            break
        clf_name = args.clf
        print "clf_name: \n", clf_name
        start = time.time()
        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print st

        if clf_name == 'gaussianmixturegd': #gaussianmixture clf_name == 'gaussianmixture'
            gaussianmixturegd_metric = np.zeros((len(param_gaussianmixturegd) + 2)).reshape(1, (len(param_gaussianmixturegd) + 2))
            print "gaussianmixturegd_metric shape: ", gaussianmixturegd_metric.shape
            for C_i, C_val in enumerate(param_gaussianmixturegd['estimator__C']):
                for batch_size_i, batch_size_val in enumerate(param_gaussianmixturegd['estimator__batch_size']):
                    for alpha_i, alpha_val in enumerate(param_gaussianmixturegd['estimator__alpha']):
                        for theta_r_lr_alpha_i, theta_r_lr_alpha_val in enumerate(param_gaussianmixturegd['estimator__theta_r_lr_alpha']):
                            for lambda_t_lr_alpha_i, lambda_t_lr_alpha_val in enumerate(param_gaussianmixturegd['estimator__lambda_t_lr_alpha']):
                                for n_gaussian_i, n_gaussian_val in enumerate(param_gaussianmixturegd['estimator__n_gaussian']):
                                    for w_init_i, w_init_val in enumerate(param_gaussianmixturegd['estimator__w_init']):
                                        for theta_alpha_i, theta_alpha_val in enumerate(param_gaussianmixturegd['estimator__theta_alpha']):
                                            for a_i, a_val in enumerate(param_gaussianmixturegd['estimator__a']):
                                                for b_i, b_val in enumerate(param_gaussianmixturegd['estimator__b']):
                                                    print "C: ", C_val
                                                    print "estimator__batch_size: ", batch_size_val
                                                    print "estimator__alpha: ", alpha_val
                                                    print "estimator__theta_r_lr_alpha: ", theta_r_lr_alpha_val
                                                    print "estimator__lambda_t_lr_alpha: ", lambda_t_lr_alpha_val
                                                    print "estimator__n_gaussian: ", n_gaussian_val
                                                    print "estimator__w_init: ", w_init_val
                                                    print "estimator__theta_alpha: ", theta_alpha_val
                                                    print "estimator__a: ", a_val
                                                    print "estimator__b: ", b_val
                                                    gaussian_mixture_gd = Gaussian_Mixture_GD_Regularization(C = C_val, batch_size = batch_size_val, alpha = alpha_val, theta_r_lr_alpha = theta_r_lr_alpha_val, lambda_t_lr_alpha = lambda_t_lr_alpha_val, n_gaussian = n_gaussian_val, w_init = w_init_val, theta_alpha = theta_alpha_val, a = a_val, b = b_val, decay = 0.01/8)
                                                    best_accuracy, best_accuracy_step = gaussian_mixture_gd.fit(X[train_index], y[train_index], X[test_index], y[test_index], 0)
                                                    print "final best_accuracy: ", best_accuracy
                                                    print "final best_accuracy_step: ", best_accuracy_step

                                                    this_model_metric = np.array([C_val, batch_size_val, alpha_val, theta_r_lr_alpha_val, lambda_t_lr_alpha_val, n_gaussian_val, w_init_val, theta_alpha_val, a_val, b_val, best_accuracy, best_accuracy_step])
                                                    this_model_metric = this_model_metric.reshape(1, this_model_metric.shape[0])
                                                    gaussianmixturegd_metric = np.concatenate((gaussianmixturegd_metric, this_model_metric), axis=0)
                                                    print "gaussianmixturegd_metric shape: ", gaussianmixturegd_metric.shape
                                                    print "gaussianmixturegd_metric: ", gaussianmixturegd_metric
            for metric_i in range(len(gaussianmixturegd_metric[:,0])):
                print gaussianmixturegd_metric[metric_i]
            print "all param best accuracy: ", np.max(gaussianmixturegd_metric[:,-2])
        elif clf_name == 'ridge':
            ridge_metric = np.zeros((len(param_ridge) + 2)).reshape(1, (len(param_ridge) + 2))
            print "ridge_metric shape: ", ridge_metric.shape

            for C_i, C_val in enumerate(param_ridge['estimator__C']):
                for lambd_i, lambd_val in enumerate(param_ridge['estimator__lambd']):
                    for batch_size_i, batch_size_val in enumerate(param_ridge['estimator__batch_size']):
                        for alpha_i, alpha_val in enumerate(param_ridge['estimator__alpha']):
                            print "C: ", C_val
                            print "estimator__lambd: ", lambd_val
                            print "estimator__batch_size: ", batch_size_val
                            print "estimator__alpha: ", alpha_val
                            ridge = Ridge_Classifier(C = C_val, lambd = lambd_val, batch_size = batch_size_val, alpha = alpha_val, decay = 0.01/8)
                            best_accuracy, best_accuracy_step = ridge.fit(X[train_index], y[train_index], X[test_index], y[test_index])
                            print "final best_accuracy: ", best_accuracy
                            print "final best_accuracy_step: ", best_accuracy_step

                            this_model_metric = np.array([C_val, lambd_val, batch_size_val, alpha_val, best_accuracy, best_accuracy_step])
                            this_model_metric = this_model_metric.reshape(1, this_model_metric.shape[0])
                            ridge_metric = np.concatenate((ridge_metric, this_model_metric), axis=0)
                            print "ridge_metric shape: ", ridge_metric.shape
                            print "ridge_metric: ", ridge_metric
            for metric_i in range(len(ridge_metric[:,0])):
                print ridge_metric[metric_i]
            # estimator__C = param_ridge['estimator__C'][random.randint(0,len(param_ridge['estimator__C'])-1)]
            # estimator__lambd = param_ridge['estimator__lambd'][random.randint(0,len(param_ridge['estimator__lambd'])-1)]
            # estimator__batch_size = param_ridge['estimator__batch_size'][random.randint(0,len(param_ridge['estimator__batch_size'])-1)]
            # estimator__alpha = param_ridge['estimator__alpha'][random.randint(0,len(param_ridge['estimator__alpha'])-1)]
            print "all param best accuracy: ", np.max(ridge_metric[:,-2])
        done = time.time()
        do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
        print do
        elapsed = done - start
        print elapsed
