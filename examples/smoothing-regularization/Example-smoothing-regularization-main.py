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
    parser.add_argument('-clf', type=str, help='classifier name')
    parser.add_argument('-categoricalindexpath', type=str, help='(optional, others are must) the categorical index path, used in NUH data set')
    parser.add_argument('-labelcolumn', type=int, help='labelcolumn, not svm')
    parser.add_argument('-batchsize', type=int, help='batchsize')
    parser.add_argument('-svmlight', type=int, help='svmlight or not')
    parser.add_argument('-sparsify', type=int, help='sparsify or not, not svm')
    parser.add_argument('-scale', type=int, help='scale or not')
    # parser.add_argument('-njob', type=int, help='multiple jobs or not')
    # parser.add_argument('-gradaverage', type=int, help='gradient average or not')

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

    print "classifier: ", args.clf

    # lasso = LogisticOneVsRestClassifier(Lasso_Classifier(batch_size=args.batchsize))
    param_lasso = {'estimator__C': [1.],
                   'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                   'estimator__batch_size': [args.batchsize],
                   'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]}

    # elastic = LogisticOneVsRestClassifier(Elasticnet_Classifier(batch_size=args.batchsize))
    param_elastic = {'estimator__C': [1.],
                     'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                     'estimator__l1_ratio': np.linspace(0.01, 0.99, 5),
                     'estimator__batch_size': [args.batchsize],
                     'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                     }

    # ridge = LogisticOneVsRestClassifier(Ridge_Classifier(batch_size=args.batchsize))
    param_ridge = {'estimator__C': [1.],
                   'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                   'estimator__batch_size': [args.batchsize],
                   'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                   }

    # huber = LogisticOneVsRestClassifier(HuberSVC(batch_size=args.batchsize))
    param_huber = {'estimator__C': [1.],
                  'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                  'estimator__mu': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                  'estimator__batch_size': [args.batchsize],
                  'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                  }

    # smoothing = LogisticOneVsRestClassifier(Smoothing_Regularization(batch_size=args.batchsize))
    param_smoothing = {'estimator__C': [1.],
                       'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                       'estimator__batch_size': [args.batchsize],
                       'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                      }

    n_folds = 5
    # param_folds = 3
    # scoring = 'accuracy'

    result_df = pandas.DataFrame()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        print "i: ", i
        if i > 0:
            break
        clf_name = args.clf
        print "clf_name: \n", clf_name
        start = time.time()
        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print st

        if clf_name == 'lasso':
            estimator__C = param_lasso['estimator__C'][random.randint(0,len(param_lasso['estimator__C'])-1)]
            estimator__lambd = param_lasso['estimator__lambd'][random.randint(0,len(param_lasso['estimator__lambd'])-1)]
            estimator__batch_size = param_lasso['estimator__batch_size'][random.randint(0,len(param_lasso['estimator__batch_size'])-1)]
            estimator__alpha = param_lasso['estimator__alpha'][random.randint(0,len(param_lasso['estimator__alpha'])-1)]
            print "C: ", estimator__C
            print "estimator__lambd: ", estimator__lambd
            print "estimator__batch_size: ", estimator__batch_size
            print "estimator__alpha: ", estimator__alpha
            lasso = Lasso_Classifier(C = estimator__C, lambd = estimator__lambd, batch_size = estimator__batch_size, alpha = estimator__alpha)
            lasso.fit(X[train_index], y[train_index], X[test_index], y[test_index])
        elif clf_name == 'ridge':
            estimator__C = param_ridge['estimator__C'][random.randint(0,len(param_ridge['estimator__C'])-1)]
            estimator__lambd = param_ridge['estimator__lambd'][random.randint(0,len(param_ridge['estimator__lambd'])-1)]
            estimator__batch_size = param_ridge['estimator__batch_size'][random.randint(0,len(param_ridge['estimator__batch_size'])-1)]
            estimator__alpha = param_ridge['estimator__alpha'][random.randint(0,len(param_ridge['estimator__alpha'])-1)]
            print "C: ", estimator__C
            print "estimator__lambd: ", estimator__lambd
            print "estimator__batch_size: ", estimator__batch_size
            print "estimator__alpha: ", estimator__alpha
            ridge = Ridge_Classifier(C = estimator__C, lambd = estimator__lambd, batch_size = estimator__batch_size, alpha = estimator__alpha)
            ridge.fit(X[train_index], y[train_index], X[test_index], y[test_index])
        elif clf_name == 'elasticnet':
            estimator__C = param_elastic['estimator__C'][random.randint(0,len(param_elastic['estimator__C'])-1)]
            estimator__lambd = param_elastic['estimator__lambd'][random.randint(0,len(param_elastic['estimator__lambd'])-1)]
            estimator__batch_size = param_elastic['estimator__batch_size'][random.randint(0,len(param_elastic['estimator__batch_size'])-1)]
            estimator__l1_ratio = param_elastic['estimator__l1_ratio'][random.randint(0,len(param_elastic['estimator__l1_ratio'])-1)]
            estimator__alpha = param_elastic['estimator__alpha'][random.randint(0,len(param_elastic['estimator__alpha'])-1)]
            print "C: ", estimator__C
            print "estimator__lambd: ", estimator__lambd
            print "estimator__batch_size: ", estimator__batch_size
            print "estimator__alpha: ", estimator__alpha
            print "estimator__l1_ratio: ", estimator__l1_ratio
            elastic = Elasticnet_Classifier(C = estimator__C, lambd = estimator__lambd, batch_size = estimator__batch_size, l1_ratio = estimator__l1_ratio, alpha = estimator__alpha)
            elastic.fit(X[train_index], y[train_index], X[test_index], y[test_index])
        elif clf_name == 'smoothing':
            estimator__C = param_smoothing['estimator__C'][random.randint(0,len(param_smoothing['estimator__C'])-1)]
            estimator__lambd = param_smoothing['estimator__lambd'][random.randint(0,len(param_smoothing['estimator__lambd'])-1)]
            estimator__batch_size = param_smoothing['estimator__batch_size'][random.randint(0,len(param_smoothing['estimator__batch_size'])-1)]
            estimator__alpha = param_smoothing['estimator__alpha'][random.randint(0,len(param_smoothing['estimator__alpha'])-1)]
            print "C: ", estimator__C
            print "estimator__lambd: ", estimator__lambd
            print "estimator__batch_size: ", estimator__batch_size
            print "estimator__alpha: ", estimator__alpha
            smoothing = Smoothing_Regularization(C = estimator__C, lambd = estimator__lambd, batch_size = estimator__batch_size, alpha = estimator__alpha)
            smoothing.fit(X[train_index], y[train_index], X[test_index], y[test_index])
        elif clf_name == 'huber':
            estimator__C = param_huber['estimator__C'][random.randint(0,len(param_huber['estimator__C'])-1)]
            estimator__lambd = param_huber['estimator__lambd'][random.randint(0,len(param_huber['estimator__lambd'])-1)]
            estimator__batch_size = param_huber['estimator__batch_size'][random.randint(0,len(param_huber['estimator__batch_size'])-1)]
            estimator__mu = param_huber['estimator__mu'][random.randint(0,len(param_huber['estimator__mu'])-1)]
            estimator__alpha = param_huber['estimator__alpha'][random.randint(0,len(param_huber['estimator__alpha'])-1)]
            print "C: ", estimator__C
            print "estimator__lambd: ", estimator__lambd
            print "estimator__batch_size: ", estimator__batch_size
            print "estimator__mu: ", estimator__mu
            print "estimator__alpha: ", estimator__alpha
            huber = HuberSVC(C = estimator__C, lambd = estimator__lambd, batch_size = estimator__batch_size, mu = estimator__mu, alpha = estimator__alpha)
            huber.fit(X[train_index], y[train_index], X[test_index], y[test_index])

#       score = accuracy_score(y[test_index], best_clf.predict(X[test_index]))
        done = time.time()
        do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
        print do
        elapsed = done - start
        print elapsed
