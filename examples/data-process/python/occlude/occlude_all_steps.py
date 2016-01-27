import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import numpy

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
test_num = int (sys.argv[1])
all_label = readData(sys.argv[2])
all_prob = readData(sys.argv[3])  #modify here
all_label_matrix = np.array(all_label[0:])[:,0:]
all_prob_matrix = np.array(all_prob[0:])[:,0:]
print all_label_matrix.shape
print all_prob_matrix.shape

print "all label length = \n", len(all_label_matrix[:,0])
true_label_prob_matrix = np.zeros([(len(all_label_matrix[:,0])/test_num), 1])
pre_AUC_matrix_test = np.zeros([(len(all_label_matrix[:,0])/test_num), 2])
print "pre_AUC_matrix_test rows = \n", (len(all_label_matrix[:,0])/test_num)


print "len(pre_AUC_matrix_test[:,0]): \n", len(pre_AUC_matrix_test[:,0])
#define model
for j in range(len(pre_AUC_matrix_test[:,0])):
    y_true, y_scores = all_label_matrix[(j*test_num):(j*test_num+test_num),0].astype(np.int), all_prob_matrix[(j*test_num):(j*test_num+test_num),0].astype(np.float) #modify here0:1018!!!!!!! less than 1018
    
    correct_0 = 0
    correct_1 = 0
    for i in range(0, test_num):
        if y_true[i] == 1 and y_scores[i] >= 0.5:
            correct_1 = correct_1 + 1
        if y_true[i] == 0 and y_scores[i] < 0.5:
            correct_0 = correct_0 + 1

    precision = (correct_0 + correct_1)/float(test_num)
    pre_AUC_matrix_test[j,0] = precision
    pre_AUC_matrix_test[j,1] = roc_auc_score(y_true[0: test_num], y_scores[0:test_num])

    sum_true_label_prob = 0
    for i in range(0, test_num):
        if y_true[i] == 1:
            sum_true_label_prob = sum_true_label_prob + y_scores[i]
        elif y_true[i] == 0:
            sum_true_label_prob = sum_true_label_prob + (1 - y_scores[i])
    true_label_prob_matrix[j,0] = sum_true_label_prob / test_num

a = numpy.asarray(true_label_prob_matrix, dtype = float)
numpy.savetxt(sys.argv[4], a, fmt = '%6f', delimiter=",") #modify here
b = numpy.asarray(pre_AUC_matrix_test, dtype = float)
numpy.savetxt(sys.argv[5], b, fmt = '%6f', delimiter=",") #modify here
#python occlude_all_steps.py 3000 ../../../NUHALLCOND/label.csv ../../../NUHALLCOND/prob.csv ../../../NUHALLCOND/true_prob_matrix.csv ../../../NUHALLCOND/pre_AUC_matrix.csv
