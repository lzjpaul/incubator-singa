import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import numpy

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
valid_num = int (sys.argv[1])
test_num = int (sys.argv[2])
all_label = readData(sys.argv[3])
all_prob = readData(sys.argv[4])  #modify here
all_label_matrix = np.array(all_label[0:])[:,0:]
all_prob_matrix = np.array(all_prob[0:])[:,0:]
print all_label_matrix.shape
print all_prob_matrix.shape

valid_test_num = valid_num + test_num
print "all label length = \n", len(all_label_matrix[:,0])
# pre_AUC_matrix_valid = np.zeros([(len(all_label_matrix[:,0])/valid_test_num), 2])
pre_AUC_matrix_test = np.zeros([(len(all_label_matrix[:,0])/3000), 2])
print "pre_AUC_matrix_valid rows = \n", (len(all_label_matrix[:,0])/3000)


print "len(pre_AUC_matrix_test[:,0]): \n", len(pre_AUC_matrix_test[:,0]) 
#define model
for j in range(len(pre_AUC_matrix_test[:,0])):
    y_true, y_scores = all_label_matrix[(j*3000):(j*3000+3000),0].astype(np.int), all_prob_matrix[(j*valid_test_num+2000):(j*valid_test_num+5000),0].astype(np.float) #modify here0:1018!!!!!!! less than 1018

    #correct_0 = 0
    #correct_1 = 0
    #for i in range(0, 0):
    #    if y_true[i] == 1 and y_scores[i] >= 0.5:
    #        correct_1 = correct_1 + 1
    #    if y_true[i] == 0 and y_scores[i] < 0.5:
    #        correct_0 = correct_0 + 1

    #precision = (correct_0 + correct_1)/float(valid_num)
    #pre_AUC_matrix_valid[j,0] = precision
    #pre_AUC_matrix_valid[j,1] = roc_auc_score(y_true[0:valid_num], y_scores[0:valid_num])
    
    correct_0 = 0
    correct_1 = 0
    for i in range(0, 3000):
        if y_true[i] == 1 and y_scores[i] >= 0.5:
            correct_1 = correct_1 + 1
        if y_true[i] == 0 and y_scores[i] < 0.5:
            correct_0 = correct_0 + 1

    precision = (correct_0 + correct_1)/float(test_num)
    pre_AUC_matrix_test[j,0] = precision
    pre_AUC_matrix_test[j,1] = roc_auc_score(y_true[valid_num: valid_test_num], y_scores[valid_num:valid_test_num])

a = numpy.asarray(pre_AUC_matrix_test, dtype = float)
numpy.savetxt(sys.argv[5], a, fmt = '%6f', delimiter=",") #modify here
b = numpy.asarray(pre_AUC_matrix_test, dtype = float)
numpy.savetxt(sys.argv[6], b, fmt = '%6f', delimiter=",") #modify here
#python aurocprecision_prob_label_sepa_all_steps.py 2000 3000 /data/zhaojing/AUC/label/version568.csv /data/zhaojing/AUC/prob/version568.csv /data/zhaojing/AUC/version568_valid_output.csv /data/zhaojing/AUC/version568_test_output.csv
