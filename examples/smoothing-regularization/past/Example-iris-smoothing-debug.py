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

import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
# from sklearn.multiclass import OneVsRestClassifier
from logistic_ovr import LogisticOneVsRestClassifier
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier
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

warnings.filterwarnings("ignore")

# data = load_iris()

# # X = data['data']
# X = scale(data['data'])
# y = data['target']
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-datapath', type=str, help='the dataset path, not svm')
    parser.add_argument('-labelpath', type=str, help='(optional, others are must) the label path, used in NUH data set, not svm')
    parser.add_argument('-categoricalindexpath', type=str, help='(optional, others are must) the categorical index path, used in NUH data set')
    parser.add_argument('-labelcolumn', type=int, help='labelcolumn, not svm')
    parser.add_argument('-batchsize', type=int, help='batchsize')
    parser.add_argument('-svmlight', type=int, help='svmlight or not')
    parser.add_argument('-sparsify', type=int, help='sparsify or not, not svm')
    parser.add_argument('-scale', type=int, help='scale or not')
    parser.add_argument('-njob', type=int, help='multiple jobs or not')
    parser.add_argument('-gradaverage', type=int, help='gradient average or not')

    args = parser.parse_args()

    labelcol = args.labelcolumn
    if args.svmlight == 1:
        X, y = svmlightclassificationDataLoader(fileName=args.datapath)
    else:
        X, y = classificationDataLoader( fileName=args.datapath, labelfile=args.labelpath, categorical_index_file = args.categoricalindexpath, labelCol=(-1 * args.labelcolumn), sparsify=(args.sparsify==1) )
    # '/data/regularization/car_evaluation/car.categorical.data')
    # /data/regularization/Audiology/audio_data/audiology.standardized.traintestcategorical.data
    print "using data loader"
    print "#process is 1?"
    print "sparsify?: ", sparse.issparse(X)

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

    lasso = LogisticOneVsRestClassifier(Lasso_Classifier(batch_size=args.batchsize))
    param_lasso = {'estimator__C': [1.],
                   'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                   'estimator__batch_size': [args.batchsize]}

    elastic = LogisticOneVsRestClassifier(Elasticnet_Classifier(batch_size=args.batchsize))
    param_elastic = {'estimator__C': [1.],
                     'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                     'estimator__l1_ratio': np.linspace(0.01, 0.99, 5),
                     'estimator__batch_size': [args.batchsize]}

    ridge = LogisticOneVsRestClassifier(Ridge_Classifier(batch_size=args.batchsize))
    param_ridge = {'estimator__C': [1.],
                   'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                   'estimator__batch_size': [args.batchsize]}

    huber = LogisticOneVsRestClassifier(HuberSVC(batch_size=args.batchsize))
    param_huber = {'estimator__C': [1.],
                  'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                  'estimator__mu': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                  'estimator__batch_size': [args.batchsize]
                  }

 #   noregulasso = LogisticOneVsRestClassifier(Lasso())
 #   param_noregulasso = {'estimator__alpha': [0]}

 #   noreguelastic = LogisticOneVsRestClassifier(ElasticNet())
 #   param_noreguelastic = {'estimator__alpha': [0],
 #                    'estimator__l1_ratio': np.linspace(0.01, 0.99, 5)}

 #   noreguridge = LogisticRidgeClassifier(solver='lsqr')
 #   param_noreguridge = {'alpha': [0]}

 #   noregu = LogisticOneVsRestClassifier(Smoothing_Regularization(batch_size=args.batchsize, gradaverage=args.gradaverage))
 #   param_noregu = {'estimator__C': [100, 10, 1, 0.1, 1e-2, 1e-3],
 #                   'estimator__lambd': [0],
 #                   'estimator__gradaverage': [args.gradaverage],
 #                   'estimator__batch_size': [args.batchsize]
                    # 'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
 #                  }

    smoothing = LogisticOneVsRestClassifier(Smoothing_Regularization(batch_size=args.batchsize))
    param_smoothing = {'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                       'estimator__batch_size': [args.batchsize]
                       #'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                      }

    n_folds = 5
    param_folds = 3
    scoring = 'accuracy'
    # scoring = 'roc_auc'

    result_df = pandas.DataFrame()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        for clf_name, clf, param_grid in [# ('Smoothing_Regularization', smoothing, param_smoothing),
                                          # ('noregu', noregu, param_noregu),
                                          # ('Lasso', lasso, param_lasso),
                                          # ('ElasticNet', elastic, param_elastic),
                                          # ('Ridge', ridge, param_ridge)
                                          # ('noregulasso', noregulasso, param_noregulasso),
                                          # ('noreguelastic', noreguelastic, param_noreguelastic),
                                          # ('noreguridge', noreguridge, param_noreguridge)
                                          ('HuberSVC', huber, param_huber)
                                          #('Lasso', lasso, param_lasso)
                                          ]:
            print "clf_name: \n", clf_name
            start = time.time()
            st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
            print st

            #number_jobs = 1
            #if args.njob == 1:
            #    number_jobs = 1
            #else:
            #    number_jobs = -1
            number_jobs = 1
            if args.njob == 100:
                number_jobs = -1
            else:
                number_jobs = args.njob
            print "number_jobs: ", number_jobs
            gs = GridSearchCV(clf, param_grid, scoring=scoring, cv=param_folds, n_jobs=number_jobs, verbose=5)
            # print "in Example-iris.py X[train] norm: ", np.linalg.norm(X[train_index])
            print "in Example-iris.py X[train]shape: ", X[train_index].shape
            print "in Example-iris.py y[train] norm: ", np.linalg.norm(y[train_index])
            print "in Example-iris.py y[train]shape: ", y[train_index].shape
            gs.fit(X[train_index], y[train_index])
            best_clf = gs.best_estimator_

            score = accuracy_score(y[test_index], best_clf.predict(X[test_index]))
            result_df.loc[i, clf_name] = score
            print 'coeficient:', best_clf.coef_, 'intercept:', best_clf.intercept_, '\n best params:', gs.best_params_, '\n best score', gs.best_score_
            done = time.time()
            do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
            print do
            elapsed = done - start
            print elapsed

    print "result shows: \n"
    result_df.loc['Mean'] = result_df.mean()
    pandas.options.display.float_format = '{:,.3f}'.format
    result_df
    print result_df.values
