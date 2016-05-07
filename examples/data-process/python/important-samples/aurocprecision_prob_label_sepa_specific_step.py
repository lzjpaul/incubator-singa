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
         
        num_label_0 = 0
        for i in range(valid_num, valid_test_num):
            if y_true[i] == 0:
                num_label_0 = num_label_0 + 1
        print "number of label 0 = \n", num_label_0
        label_0_probs = np.zeros([num_label_0, 2]) # the prob of label 0 samples
        label_1_probs = np.zeros([test_num - num_label_0, 2]) # (sample_num, prob)
        label_0_idx = 0
        label_1_idx = 0
        for i in range(valid_num, valid_test_num):
            if y_true[i] == 0:
                label_0_probs[label_0_idx, 0] = i - valid_num
                label_0_probs[label_0_idx, 1] = y_scores[i]
                label_0_idx = label_0_idx + 1
            if y_true[i] == 1:
                label_1_probs[label_1_idx, 0] = i - valid_num
                label_1_probs[label_1_idx, 1] = y_scores[i]
                label_1_idx = label_1_idx + 1
        print "final label_0_idx = \n", label_0_idx
        print "final label_1_idx = \n", label_1_idx
        print "\n\n\n"
         
        # need to check (1) label is 0 or 1? (2) sum up to 2910?? 
        print "top 20 correct_0_index = \n", label_0_probs[:, 0][np.argsort(label_0_probs[:, 1])][0:print_top_k]
        print "top 20 correct_0_prob = \n", label_0_probs[:, 1][np.argsort(label_0_probs[:, 1])][0:print_top_k]
        print "\n\n\n"
        print "top 20 incorrect_0_index = \n", label_0_probs[:, 0][np.argsort(-label_0_probs[:, 1])][0:print_top_k]
        print "top 20 incorrect_0_prob = \n", label_0_probs[:, 1][np.argsort(-label_0_probs[:, 1])][0:print_top_k]
        print "\n\n\n"

        print "top 20 correct_1_index = \n", label_1_probs[:, 0][np.argsort(-label_1_probs[:, 1])][0:print_top_k]
        print "top 20 correct_1_prob = \n", label_1_probs[:, 1][np.argsort(-label_1_probs[:, 1])][0:print_top_k]
        print "\n\n\n"
        print "top 20 incorrect_1_index = \n", label_1_probs[:, 0][np.argsort(label_1_probs[:, 1])][0:print_top_k]
        print "top 20 incorrect_1_prob = \n", label_1_probs[:, 1][np.argsort(label_1_probs[:, 1])][0:print_top_k]
        print "\n\n\n"
         
        # will affect y_scores array?
        y_scores_05_distance = np.zeros(len(y_scores))
        for i in range(len(y_scores)):
            y_scores_05_distance[i] = abs(y_scores[i] - 0.5)
        print "top 20 test vague_index = \n", np.argsort(y_scores_05_distance[valid_num:valid_test_num])[0:print_top_k]
        print "top 20 test vague_prob = \n", y_scores[valid_num:valid_test_num][np.argsort(y_scores_05_distance[valid_num:valid_test_num])][0:print_top_k]
        print "\n\n\n"

        pre_AUC_matrix_test[j,1] = roc_auc_score(y_true[valid_num: valid_test_num], y_scores[valid_num:valid_test_num])
        print "test AUC = \n", roc_auc_score(y_true[valid_num: valid_test_num], y_scores[valid_num:valid_test_num])
        a = numpy.asarray(y_scores[valid_num:valid_test_num], dtype = float)
        numpy.savetxt("test_y_scores_for_check.csv", a, fmt = '%6f', delimiter=",") #modify here
        break

a = numpy.asarray(label_0_probs[:, 0][np.argsort(label_0_probs[:, 1])][0:print_top_k], dtype = int)
numpy.savetxt(sys.argv[7], a, fmt = '%d', delimiter=",") #modify here
b = numpy.asarray(label_1_probs[:, 0][np.argsort(-label_1_probs[:, 1])][0:print_top_k], dtype = int)
numpy.savetxt(sys.argv[8], b, fmt = '%d', delimiter=",") #modify here
c = numpy.asarray(np.argsort(y_scores_05_distance[valid_num:valid_test_num])[0:print_top_k], dtype = int)
numpy.savetxt(sys.argv[9], c, fmt = '%d', delimiter=",") #modify here
d = numpy.asarray(label_0_probs[:, 0][np.argsort(-label_0_probs[:, 1])][0:print_top_k], dtype = int)
numpy.savetxt(sys.argv[10], d, fmt = '%d', delimiter=",") #modify here
e = numpy.asarray(label_1_probs[:, 0][np.argsort(label_1_probs[:, 1])][0:print_top_k], dtype = int)
numpy.savetxt(sys.argv[11], e, fmt = '%d', delimiter=",") #modify here
# python aurocprecision_prob_label_sepa_specific_step.py 1940 2910 1-16-1SRC-KB/label.csv 1-16-1SRC-KB/prob.csv 512 20 1-16-1SRC-KB/correct_label_0_index.csv 1-16-1SRC-KB/correct_label_1_index.csv 1-16-1SRC-KB/vague_index.csv 1-16-1SRC-KB/incorrect_label_0_index.csv 1-16-1SRC-KB/incorrect_label_1_index.csv
# python aurocprecision_prob_label_sepa_specific_step.py 1940 2910 2-5-CNN-PEARSON-3-best-models/version8199/label.csv 2-5-CNN-PEARSON-3-best-models/version8199/prob.csv 175 20 correct_label_0_index.csv correct_label_1_index.csv vague_index.csv
