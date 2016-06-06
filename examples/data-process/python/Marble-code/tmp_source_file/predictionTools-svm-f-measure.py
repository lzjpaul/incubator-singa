import itertools
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
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
testX_all = np.genfromtxt(file, delimiter=",")
testX = testX_all
print "testX shape = \n", testX.shape

# test_label = readData("/data/zhaojing/marble/Marble_08_09_Car_Cla_UMLS_test_label.csv")
file = open(sys.argv[4])
testY_all = np.genfromtxt(file, delimiter=",")
print "testY_all shape = \n", testY_all.shape
testY = testY_all
print "testY shape = \n", testY.shape

model = SVC(kernel='rbf', probability=True)
model.fit(trainX, trainY.T)
modelPred = model.predict_proba(testX)
fpr, tpr, thresholds = metrics.roc_curve(testY.T, modelPred[:, 1], pos_label=1)
auc = metrics.auc(fpr, tpr)
print "auc = \n", auc


y_true, y_scores = testY.T.astype(np.int), modelPred[:, 1].astype(np.float) #modify here0:1018!!!!!!! less than 1018
# print "y_true shape = \n", y_true.shape
# print "y_scores shape = \n", y_scores.shape
# print "y_true = \n", y_true
# print "y_scores = \n", y_scores
# print "y_scores max = \n", max(y_scores)
# print "y_scores max index = \n", np.argsort(y_scores)
# print "max y_scores = \n", y_scores[2539]
# print "max y_score label = \n", y_true[2539]
count = y_scores.shape[0]
true_0 = 0
true_1 = 0
predict_0 = 0
predict_1 = 0
correct_0 = 0
correct_1 = 0
for i in range(0, count):
    # print "y_scores"
    # print y_scores[i]
    if y_true[i] == 1:
        true_1 = true_1 + 1
    else:
        true_0 = true_0 + 1

    if y_scores[i] >= 0.5:
        predict_1 = predict_1 + 1
    else:
        predict_0 = predict_0 + 1

    if y_true[i] == 1 and y_scores[i] >= 0.5:
        correct_1 = correct_1 + 1
    if y_true[i] == 0 and y_scores[i] < 0.5:
        correct_0 = correct_0 + 1

#check logic
print "predict_1 = \n", predict_1
print "correct_1 = \n", correct_1
#check logic


print "Metric Begin"

precision_1 = correct_1 / float(predict_1)
print "precision_1"
print precision_1
recall_1 = correct_1 / float(true_1)
print "recall_1"
print recall_1

Fmeasure_1 = 2*precision_1*recall_1 / float(precision_1 + recall_1)
print "F-measure_1"
print Fmeasure_1

precision_0 = correct_0/ float(predict_0)
print "precision_0"
print precision_0
recall_0 = correct_0/ float(true_0)
print "recall_0"
print recall_0
Fmeasure_0 = 2*precision_0*recall_0 / float(precision_0 + recall_0)
print "F_measure_0"
print Fmeasure_0


#auroc score
print "roc"
print roc_auc_score(y_true, y_scores)

print "count"
print count

precision = (correct_0 + correct_1)/float(count)
print "overall precision"
print precision

# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_S1_50_3.txt
# /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_S1_50_3.txt
# /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
# python predictionTools.py /data/zhaojing/marble/tensor/dataset1/Marble_split_train_label_1.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_train_S1_50_3.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult-10000/membershipresult/membership_test_S1_50_3.txt /data/zhaojing/marble/tensor/dataset1/Marble_split_test_label_1.csv
