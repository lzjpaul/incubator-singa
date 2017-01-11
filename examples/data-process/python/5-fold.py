# 1-7: only the smoothing, copying from lasso, param from 100

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
#from huber_svm import HuberSVC
#from lasso_clf import Lasso_Classifier
#from ridge_clf import Ridge_Classifier
#from elasticnet_clf import Elasticnet_Classifier
#from smoothing_regularization import Smoothing_Regularization

# import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
# from sklearn.multiclass import OneVsRestClassifier
# from logistic_ovr import LogisticOneVsRestClassifier
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
# from sklearn.preprocessing import scale
from sklearn import preprocessing
#from DataLoader import classificationDataLoader
#from svmlightDataLoader import svmlightclassificationDataLoader
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
    parser.add_argument('-outputpath', type=str, help='outputpath')

    args = parser.parse_args()


    X = np.genfromtxt(args.datapath, delimiter = ',')
    y = np.genfromtxt(args.labelpath, delimiter = ',')
    print "X shape: ", X.shape
    print "y shape: ", y.shape
    np.random.seed(10)
    idx = np.random.permutation(X.shape[0])
    print "idx: ", idx
    X = X[idx]
    y = y[idx]

    n_folds = 5

    for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        print "i: ", i
        a = np.asarray(X[train_index], dtype = int)
        b = np.asarray(y[train_index], dtype = int)
        print "traindata shape: ", a.shape
        np.savetxt(args.outputpath + '/traindata' + str(i) + '.csv', a, fmt = '%d', delimiter=",") #modify here
        np.savetxt(args.outputpath + '/trainlabel' + str(i) + '.csv', b, fmt = '%d', delimiter=",") #modify here
        a = np.asarray(X[test_index], dtype = int)
        b = np.asarray(y[test_index], dtype = int)
        print "testdata shape: ", a.shape
        np.savetxt(args.outputpath + '/testdata' + str(i) + '.csv', a, fmt = '%d', delimiter=",") #modify here
        np.savetxt(args.outputpath + '/testlabel' + str(i) + '.csv', b, fmt = '%d', delimiter=",") #modify here
