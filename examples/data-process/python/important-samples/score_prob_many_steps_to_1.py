# if step 47 is the best step, fetch step 47 prob and label matrix
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
pre_AUC_matrix_valid = np.zeros([(len(all_label_matrix[:,0])/valid_test_num), 2])
pre_AUC_matrix_test = np.zeros([(len(all_label_matrix[:,0])/valid_test_num), 2])
print "pre_AUC_matrix_valid rows = \n", (len(all_label_matrix[:,0])/valid_test_num)


print "len(pre_AUC_matrix_test[:,0]): \n", len(pre_AUC_matrix_test[:,0])

specific_step = int (sys.argv[5])
print_top_k = int (sys.argv[6]) # top 20
#define model
for j in range(len(pre_AUC_matrix_test[:,0])):
    if j != specific_step:
        continue
    else:
        print "specific step = \n", specific_step
        y_true, y_scores = all_label_matrix[(j*valid_test_num):(j*valid_test_num+valid_test_num),0].astype(np.int), all_prob_matrix[(j*valid_test_num):(j*valid_test_num+valid_test_num),0].astype(np.float) #modify here0:1018!!!!!!! less than 1018
        a = numpy.asarray(y_scores[valid_num:valid_test_num], dtype = float)
        numpy.savetxt(sys.argv[7], a, fmt = '%6f', delimiter=",") #modify here
        break
#python score_prob_many_steps_to_1.py 1940 2910 1-16-1SRC-KB/label.csv 1-16-1SRC-KB/prob.csv 512 20 1-16-1SRC-KB/correct_label_0_index.csv 1-16-1SRC-KB/1step_prob.csv
# python aurocprecision_prob_label_sepa_specific_step.py 1940 2910 1-16-1SRC-KB/label.csv 1-16-1SRC-KB/prob.csv 512 20 1-16-1SRC-KB/correct_label_0_index.csv 1-16-1SRC-KB/correct_label_1_index.csv 1-16-1SRC-KB/vague_index.csv 1-16-1SRC-KB/incorrect_label_0_index.csv 1-16-1SRC-KB/incorrect_label_1_index.csv
# python aurocprecision_prob_label_sepa_specific_step.py 1940 2910 2-5-CNN-PEARSON-3-best-models/version8199/label.csv 2-5-CNN-PEARSON-3-best-models/version8199/prob.csv 175 20 correct_label_0_index.csv correct_label_1_index.csv vague_index.csv
