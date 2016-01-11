import numpy as np
from sklearn.metrics import roc_auc_score
import sys

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
begin_index = int (sys.argv[1])
end_index = int (sys.argv[2])
train_label = readData(sys.argv[3])
train_label_matrix = np.array(train_label[begin_index:end_index])[:,0:]
train_prob = readData(sys.argv[4])  #modify here
train_prob_matrix = np.array(train_prob[begin_index:end_index])[:,0:]


#define model
print train_label_matrix.shape
print train_prob_matrix.shape
y_true, y_scores = train_label_matrix[:,0].astype(np.int), train_prob_matrix[:,0].astype(np.float) #modify here0:1018!!!!!!! less than 1018
#print y_scores[0]
#print y_scores[1]
#print y_scores[2]
#print y_scores[3]
#print "y_scores shape"
#print y_scores.shape[0]
count = y_scores.shape[0]
print "count"
print count
true_0 = 0
true_1 = 0
predict_0 = 0
predict_1 = 0
correct_0 = 0
correct_1 = 0
for i in range(0, count):
    #print "y_scores"
    #print y_scores[i]
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

#print "true_1"
#print true_1
#print "true_0"
#print true_0
#print "predict_1"
#print predict_1
#print "predict_0"
#print predict_0
#print "correct_1"
#print correct_1
#print "correct_0"
#print correct_0

print "Metric Begin"
precision = (correct_0 + correct_1)/float(count)
print "overall precision"
print precision

# precision_1 = correct_1 / float(predict_1)
print "precision_1"
# print precision_1
recall_1 = correct_1 / float(true_1)
print "recall_1"
print recall_1

# Fmeasure_1 = 2*precision_1*recall_1 / float(precision_1 + recall_1)
print "F-measure_1"
# print Fmeasure_1

# precision_0 = correct_0/ float(predict_0)
print "precision_0"
# print precision_0
recall_0 = correct_0/ float(true_0)
print "recall_0"
print recall_0
# Fmeasure_0 = 2*precision_0*recall_0 / float(precision_0 + recall_0)
print "F_measure_0"
# print Fmeasure_0


#auroc score
print "roc"
print roc_auc_score(y_true, y_scores)

#python aurocprecision_prob_label_sepa.py 21000 24000 /data/zhaojing/AUC/label/version285.csv /data/zhaojing/AUC/prob/version285.csv
