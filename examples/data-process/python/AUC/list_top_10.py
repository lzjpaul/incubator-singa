import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import numpy
import os

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

def list_top_10(test_output):
    #read training data
    #valid_num = int (sys.argv[1])
    #test_num = int (sys.argv[2])
    test_output_data = readData(test_output)
    test_output_matrix = np.array(test_output_data[0:])[:,0:]
    print test_output_matrix.shape

    # test_AUC_vector = test_output_matrix[:,1]
    test_AUC_vector = test_output_matrix[:,1].T
    print "test_AUC_vector shape = \n", test_AUC_vector.shape
    # test_AUC_vector = -test_AUC_vector
    # print "top 10 sort order1 = = \n", np.argsort(test_AUC_vector)
    # print "top 10 sort order = \n", np.argsort(test_output_matrix[:,1])
    print "top 10 = \n", test_AUC_vector[np.argsort(test_output_matrix[:,1])][590:600]

#python aurocprecision_prob_label_sepa_all_steps.py 2000 3000 /data/zhaojing/AUC/label/version568.csv /data/zhaojing/AUC/prob/version568.csv /data/zhaojing/AUC/version568_valid_output.csv /data/zhaojing/AUC/version568_test_output.csv

arr_test_output = []

list_dirs = os.walk(sys.argv[1]) #test

for root, dirs, files in list_dirs:
    for f in files:
        print os.path.join(root, f)
        arr_test_output.append(os.path.join(root, f))

for i in range(len(arr_test_output)):
    print "processing = \n", arr_test_output[i]
    list_top_10(arr_test_output[i])

#python list_top_10.py /data/zhaojing/AUC/test_result/
