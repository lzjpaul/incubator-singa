import itertools
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn import cross_validation
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

model = SVC(kernel='rbf', probability=True)
# model.fit(trainX, trainY.T)
# cross validation
scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='roc_auc')
print "roc_auc: ", scores
scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='accuracy')
print "accuracy: ", scores
scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='average_precision')
print "average_precision: ", scores
scores = cross_validation.cross_val_score(model, trainX, trainY, cv=5, scoring='f1')
print "f1: ", scores




# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_S1_50_3.txt
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_S1_50_3.txt
# /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_train_S1_50_3.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_test_S1_50_3.txt /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
