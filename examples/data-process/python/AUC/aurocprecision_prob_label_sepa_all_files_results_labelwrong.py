import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import numpy
import os
import random
from random import randint

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

def calAUC(valid_num, test_num, all_label_file, all_prob_file, valid_output, test_output):
    #read training data
    #valid_num = int (sys.argv[1])
    #test_num = int (sys.argv[2])
    all_label = readData(all_label_file)
    all_prob = readData(all_prob_file)  #modify here
    all_label_matrix = np.array(all_label[0:])[:,0:]
    all_prob_matrix = np.array(all_prob[0:])[:,0:]
    # print all_label_matrix.shape
    # print all_prob_matrix.shape

    valid_test_num = valid_num + test_num
    # print "all label length = \n", len(all_label_matrix[:,0])
    pre_AUC_matrix_valid = np.zeros([(len(all_label_matrix[:,0])/valid_test_num), 2])
    pre_AUC_matrix_test = np.zeros([(len(all_label_matrix[:,0])/valid_test_num), 2])
    # print "pre_AUC_matrix_valid rows = \n", (len(all_label_matrix[:,0])/valid_test_num)


    # print "len(pre_AUC_matrix_test[:,0]): \n", len(pre_AUC_matrix_test[:,0]) 
    #define model
    for j in range(len(pre_AUC_matrix_test[:,0])):
        y_true, y_scores = all_label_matrix[(j*valid_test_num):(j*valid_test_num+valid_test_num),0].astype(np.int), all_prob_matrix[(j*valid_test_num):(j*valid_test_num+valid_test_num),0].astype(np.float) #modify here0:1018!!!!!!! less than 1018

        correct_0 = 0
        correct_1 = 0
        for i in range(0, valid_num):
            if y_true[i] == 1 and y_scores[i] >= 0.5:
                correct_1 = correct_1 + 1
            if y_true[i] == 0 and y_scores[i] < 0.5:
                correct_0 = correct_0 + 1

        precision = (correct_0 + correct_1)/float(valid_num)
        pre_AUC_matrix_valid[j,0] = precision
        pre_AUC_matrix_valid[j,1] = roc_auc_score(y_true[0:valid_num], y_scores[0:valid_num])
        
        correct_0 = 0
        correct_1 = 0
        for i in range(valid_num, valid_test_num):
            if y_true[i] == 1 and y_scores[i] >= 0.5:
                correct_1 = correct_1 + 1
            if y_true[i] == 0 and y_scores[i] < 0.5:
                correct_0 = correct_0 + 1

        precision = (correct_0 + correct_1)/float(test_num)
        pre_AUC_matrix_test[j,0] = precision
        pre_AUC_matrix_test[j,1] = roc_auc_score(y_true[valid_num: valid_test_num], y_scores[valid_num:valid_test_num])

    a = numpy.asarray(pre_AUC_matrix_valid, dtype = float)
    numpy.savetxt(valid_output, a, fmt = '%6f', delimiter=",") #modify here
    b = numpy.asarray(pre_AUC_matrix_test, dtype = float)
    numpy.savetxt(test_output, b, fmt = '%6f', delimiter=",") #modify here
    best_validataion_place = np.argsort(pre_AUC_matrix_valid[:,1])[len(pre_AUC_matrix_valid[:,1]) - 1]
    print "best validation place = \n", best_validataion_place
    print "best test AUC = \n", pre_AUC_matrix_test[best_validataion_place,1]
    return best_validataion_place, pre_AUC_matrix_test[best_validataion_place,1]
    #python aurocprecision_prob_label_sepa_all_steps.py 2000 3000 /data/zhaojing/AUC/label/version568.csv /data/zhaojing/AUC/prob/version568.csv /data/zhaojing/AUC/version568_valid_output.csv /data/zhaojing/AUC/version568_test_output.csv

arr_label = []
arr_prob = []
best_AUC = []

list_dirs = os.walk(sys.argv[1]) #label

for root, dirs, files in list_dirs:
    for f in files:
        # print os.path.join(root, f)
        if 'label' in os.path.join(root, f):
            arr_label.append(os.path.join(root, f))
arr_label.sort()

list_dirs = os.walk(sys.argv[1]) #prob
for root, dirs, files in list_dirs:
    for f in files:
        # print os.path.join(root, f)
        if 'prob' in os.path.join(root, f):
            arr_prob.append(os.path.join(root, f))
arr_prob.sort()
# for i in range(len(arr_label)):
#    print "label file = \n", arr_label[i]
#    print "prob file = \n", arr_prob[i]

f = open(sys.argv[1] +'infoversion' + str(random.randint(0,20000)), 'w+')
for i in range(len(arr_label)):
    print "i is = ", i
    f.write("i is = " + str(i) + "\n")
    print "processing = ", arr_label[i]
    f.write("processing = " + arr_label[i] + "\n")
    print "processing = ", arr_prob[i]
    f.write("processing = " + arr_prob[i] + "\n")
    best_validation_place, best_AUC_value = calAUC(2000, 3000, arr_label[i], arr_prob[i], sys.argv[1] + "valid_" + str(i) + ".csv", sys.argv[1] + "test_" + str(i) + ".csv")
    best_AUC.append(best_AUC_value)
    f.write("best_AUC_value = " + str(best_AUC_value) + "\n")
    f.write("best_validation_place = " + str(best_validation_place) + "\n")
best_AUC = np.array(best_AUC)
print "top 10 model = ", np.argsort(-best_AUC)[0:10]
print "top 10 AUC = ", best_AUC[np.argsort(-best_AUC)][0:10]
for i in range(10):
    f.write("i = " + str(i) + "\n")
    f.write("top 10 model index = " + str(np.argsort(-best_AUC)[i]) + "\n")
    f.write("top 10 AUC = " +  str(best_AUC[np.argsort(-best_AUC)][i]) + "\n")
f.close()
# python aurocprecision_prob_label_sepa_all_files_results.py /data/zhaojing/result/
