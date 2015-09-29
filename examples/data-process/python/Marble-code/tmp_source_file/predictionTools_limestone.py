import itertools
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

# train_label = readData("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_train_label.csv")
file = open("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_train_label.csv")
trainY = np.genfromtxt(file, delimiter=",")
print "trainY shape = \n", trainY.shape

# train_data = readData("/data/zhaojing/marble/membership_train_50_5.txt")
file = open("/data/zhaojing/marble/lstone_membership_train.txt")

trainX = np.genfromtxt(file, delimiter=",")
print "trainX shape = \n", trainX.shape

# test_data = readData("/data/zhaojing/marble/membership_test_50_5.txt")
file = open("/data/zhaojing/marble/lstone_membership_test.txt")
testX = np.genfromtxt(file, delimiter=",")
print "testX shape = \n", testX.shape

# test_label = readData("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_test_label.csv")
file = open("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_test_label.csv")
testY = np.genfromtxt(file, delimiter=",")
print "testY shape = \n", testY.shape

model = LogisticRegression(penalty = 'l1', tol=0.0001)
model.fit(trainX, trainY.T)
modelPred = model.predict_proba(testX)
fpr, tpr, thresholds = metrics.roc_curve(testY.T, modelPred[:, 1], pos_label=1)
auc = metrics.auc(fpr, tpr)
print "auc = \n", auc
