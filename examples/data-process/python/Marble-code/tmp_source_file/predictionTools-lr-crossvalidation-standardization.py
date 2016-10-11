import itertools
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import preprocessing
import math
from math import pow
import sys


def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

# train_label = readData("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_train_label.csv")
file = open(sys.argv[2])
trainY = np.genfromtxt(file, delimiter=",")
print "trainY shape = \n", trainY.shape

# train_data = readData("/data/zhaojing/marble/membership_train_50_5.txt")
file = open(sys.argv[1])

trainX = np.genfromtxt(file, delimiter=",")
print "trainX shape = \n", trainX.shape

standardization_flag = int(sys.argv[3]) # 1: standardization, 0: not standardization
if standardization_flag == 1:
    print "need standardization!!\n"
    scaler = preprocessing.StandardScaler().fit(trainX)
    trainX = scaler.transform(trainX)

regularization_pow = int(sys.argv[4])
regularization_term = float(math.pow(10, -1*regularization_pow))

print "regularization_term: \n", regularization_term

model = LogisticRegression(penalty = 'l2', tol=0.0001, C=regularization_term)
# model.fit(trainX, trainY.T)
# cross validation
scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='roc_auc')
print "roc_auc: ", scores
scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='accuracy')
print "accuracy: ", scores
# scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='average_precision')
# print "precision: ", scores
scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='f1')
print "f1: ", scores

# python predictionTools-lr-crossvalidation-standardization.py /ssd/zhaojing/cnn/NUHALLCOND/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_data_1case.csv /ssd/zhaojing/cnn/NUHALLCOND/VISIT_DIAG_aggcnt_1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_label_1case.csv 1 0
# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_S1_50_3.txt
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_S1_50_3.txt
# /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_train_S1_50_3.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_test_S1_50_3.txt /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
