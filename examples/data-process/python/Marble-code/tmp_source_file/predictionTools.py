import itertools
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import sys


def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

# train_label = readData("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_train_label.csv")
file = open(sys.argv[1])
trainY = np.genfromtxt(file, delimiter=",")
print "trainY shape = \n", trainY.shape

# train_data = readData("/data/zhaojing/marble/membership_train_50_5.txt")
file = open(sys.argv[2])

trainX = np.genfromtxt(file, delimiter=",")
print "trainX shape = \n", trainX.shape

# test_data = readData("/data/zhaojing/marble/membership_test_50_5.txt")
file = open(sys.argv[3])
testX = np.genfromtxt(file, delimiter=",")
print "testX shape = \n", testX.shape

# test_label = readData("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_test_label.csv")
file = open(sys.argv[4])
testY = np.genfromtxt(file, delimiter=",")
print "testY shape = \n", testY.shape

model = LogisticRegression(penalty = 'l1', tol=0.000001)
model.fit(trainX, trainY.T)
modelPred = model.predict_proba(testX)
fpr, tpr, thresholds = metrics.roc_curve(testY.T, modelPred[:, 1], pos_label=1)
auc = metrics.auc(fpr, tpr)
print "auc = \n", auc

# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_S1_50_3.txt
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_S1_50_3.txt
# /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_train_S1_50_3.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_test_S1_50_3.txt /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
