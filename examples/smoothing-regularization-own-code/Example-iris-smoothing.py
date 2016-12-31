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
from smoothing_regularization import Smoothing_Regularization

import pandas
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
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
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    param_lasso = {'estimator__alpha': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]}

    param_elastic = {'estimator__alpha': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4], 
                     'estimator__l1_ratio': np.linspace(0.01, 0.99, 5)}

    ridge = RidgeLR(batch_size=args.batchsize)
    param_ridge = {'alpha': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]}

    huber = HuberLR(batch_size=args.batchsize)
    param_huber = {#'estimator__C': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                  'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4], 
                  'estimator__mu': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]}

    #noregulasso = OneVsRestClassifier(Lasso())
    #param_noregulasso = {'estimator__alpha': [0]}

    #noreguelastic = OneVsRestClassifier(ElasticNet())
    #param_noreguelastic = {'estimator__alpha': [0], 
    #                'estimator__l1_ratio': np.linspace(0.01, 0.99, 5)}

    #noreguridge = RidgeClassifier(solver='lsqr')
    #param_noreguridge = {'alpha': [0]}
    #
    #noregu = OneVsRestClassifier(Smoothing_Regularization(batch_size=args.batchsize, gradaverage=args.gradaverage))
    #param_noregu = {'estimator__C': [100, 10, 1, 0.1, 1e-2, 1e-3],
    #                'estimator__lambd': [0],
    #                'estimator__gradaverage': [args.gradaverage],
    #                'estimator__batch_size': [args.batchsize]
    #                # 'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
    #              }

    smoothing = OneVsRestClassifier(Smoothing_Regularization(batch_size=args.batchsize))
    param_smoothing = {#'estimator__C': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                       'estimator__lambd': [1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                       #'estimator__alpha': [10000, 1000, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4]
                      }

    n_folds = 5
    param_folds = 3
    scoring1 = 'accuracy'
    scoring2 = 'AUC'

    accuracy_result_df = pandas.DataFrame()
    auc_result_df = pandas.DataFrame()
    for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        inner_train_index_list = []
        inner_test_index_list = []
        #???????????should be here????????????????? data shape correct??/
        # inner train and validation
        for inner_i, (inner_train_index, inner_test_index) in enumerate(StratifiedKFold(y[train_index], n_folds=param_folds)):
            inner_train_index_list.append(inner_train_index)
            inner_test_index_list.append(inner_test_index)
        for clf_name, param_grid in [('Smoothing_Regularization', param_smoothing),
                                          # ('noregu', noregu, param_noregu),
                                          ('Lasso', param_lasso),
                                          ('ElasticNet', param_elastic),
                                          ('Ridge', param_ridge)
                                          # ('noregulasso', noregulasso, param_noregulasso),
                                          # ('noreguelastic', noreguelastic, param_noreguelastic), 
                                          # ('noreguridge', noreguridge, param_noreguridge)
                                          #('HuberSVC', huber, param_huber)
                                          #('Lasso', lasso, param_lasso)
                                          ]:
            #number_jobs = 1
            #if args.njob == 100:
            #    number_jobs = -1
            #else:
            #    number_jobs = args.njob
            #print "number_jobs: ", number_jobs
            #gs = GridSearchCV(clf, param_grid, scoring=scoring, cv=param_folds, n_jobs=number_jobs)

            #gs.fit(X[train_index], y[train_index])
            #best_clf = gs.best_estimator_
            if clf_name == 'Lasso':
                print "clf_name: \n", clf_name
                start = time.time()
                st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                print st

                for estimator__alpha_i in range(len(param_lasso['estimator__alpha'])):
                    selected_estimator_alpha = param_lasso['estimator__alpha'][estimator__alpha_i]
                    param_accuracy_avg_best = 0.0
                    best_estimator__alpha = 1000
                    param_accuracy = []
                    for validation_i in range(param_folds):
                        #???? every time a new model, or can just put above, every parameter a new model????
                        lasso = LassoLR(batch_size=args.batchsize, estimator__alpha = selected_estimator_alpha)
                        lasso.fit(X[train_index][inner_train_index_list[validation_i]], y[train_index][inner_train_index_list[validation_i]])
                        param_accuracy.append(accuracy_score(y[test_index][inner_test_index_list[validation_i]], lasso.predict(X[test_index])))
                    #should be float here !!!??????
                    param_accuracy_avg = sum(param_accuracy) / float(len(param_accuracy))
                    if (param_accuracy_avg > param_accuracy_avg_best):
                        param_accuracy_avg_best = param_accuracy_avg
                        best_estimator__alpha = selected_estimator_alpha

                lasso = LassoLR(batch_size=args.batchsize, estimator__alpha = best_estimator__alpha)
                lasso.fit(X[train_index], y[train_index])
                score1 = accuracy_score(y[test_index], lasso.predict(X[test_index]))
                score2 = roc_auc_score(y[test_index], lasso.predict_proba(X[test_index]))
                accuracy_result_df.loc[i, clf_name] = score1
                auc_result_df.loc[i, clf_name] = score2
                done = time.time()
                do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                print do
                elapsed = done - start
                print elapsed

            elif clf_name == 'ElasticNet':
                print "clf_name: \n", clf_name
                start = time.time()
                st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                print st

                for estimator__alpha_i in range(len(param_elastic['estimator__alpha'])):
                    for estimator__l1_ratio_i in range(len(param_elastic['estimator__l1_ratio'])):
                        selected_estimator_alpha = param_elastic['estimator__alpha'][estimator__alpha_i]
                        selected_l1_ratio = param_elastic['estimator__l1_ratio'][estimator__l1_ratio_i]
                        param_accuracy_avg_best = 0.0
                        best_estimator__alpha = 1000
                        best_estimator__l1_ratio = 0.01
                        param_accuracy = []
                        for validation_i in range(param_folds):
                            #???? every time a new model, or can just put above, every parameter a new model????
                            elastic = ElasticNetLR(batch_size=args.batchsize(), estimator__alpha = selected_estimator_alpha, estimator__l1_ratio = selected_l1_ratio)
                            elastic.fit(X[train_index][inner_train_index_list[validation_i]], y[train_index][inner_train_index_list[validation_i]])
                            param_accuracy.append(accuracy_score(y[test_index][inner_test_index_list[validation_i]], elastic.predict(X[test_index])))
                        #should be float here !!!??????
                        param_accuracy_avg = sum(param_accuracy) / float(len(param_accuracy))
                        if (param_accuracy_avg > param_accuracy_avg_best):
                            param_accuracy_avg_best = param_accuracy_avg
                            best_estimator__alpha = selected_estimator_alpha
                            best_estimator__l1_ratio = selected_l1_ratio

                # best here!!!??? can print out the selected and best parameters
                elastic = ElasticNetLR(batch_size=args.batchsize, estimator__alpha = best_estimator__alpha, estimator__l1_ratio = best_l1_ratio)
                elastic.fit(X[train_index], y[train_index])
                score1 = accuracy_score(y[test_index], elastic.predict(X[test_index]))
                score2 = roc_auc_score(y[test_index], elastic.predict_proba(X[test_index]))
                accuracy_result_df.loc[i, clf_name] = score1
                auc_result_df.loc[i, clf_name] = score2
                done = time.time()
                do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                print do
                elapsed = done - start
                print elapsed
            
            #print 'coeficient:', best_clf.coef_, 'intercept:', best_clf.intercept_, '\n best params:', gs.best_params_, '\n best score', gs.best_score_
           

    print "result shows: \n"
    result_df.loc['Mean'] = result_df.mean()
    pandas.options.display.float_format = '{:,.3f}'.format
    result_df
    print result_df.values
