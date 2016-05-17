'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import random
import sys


batch_size = 100 #modify for test and valid
nb_classes = 2
nb_epoch = 20

#read data
file = open(sys.argv[1])
train_data_matrix = np.genfromtxt(file, delimiter=",")
train_data_matrix = train_data_matrix.astype('float32')
file.close()
print ("train_data_matrix = ", train_data_matrix.shape)

#read data
file = open(sys.argv[2])
train_label_matrix = np.genfromtxt(file, delimiter=",")
train_label_matrix = train_label_matrix.astype(np.int)
file.close()
print ("train_label_matrix = ", train_label_matrix.shape)

#read data
file = open(sys.argv[3])
test_data_matrix = np.genfromtxt(file, delimiter=",")
test_data_matrix = test_data_matrix.astype('float32')
file.close()
print ("test_data_matrix = ", test_data_matrix.shape)

#read data
file = open(sys.argv[4])
test_label_matrix = np.genfromtxt(file, delimiter=",")
test_label_matrix = test_label_matrix.astype(np.int)
file.close()
print ("test_label_matrix = ", test_label_matrix.shape)

# the data, shuffled and split between train and test sets
case_num = 12
case_feature_dim = 1253

train_data_matrix = train_data_matrix.reshape(4850, case_num, case_feature_dim)
test_data_matrix = test_data_matrix.reshape(2910, case_num, case_feature_dim)
print ("before max of first case: ", max(train_data_matrix[0,0,:]))

last_case_idx = 0
for i in range(4850):
    # print("i is: ", i)
    last_case_idx = case_num # defaultly regard as the last one
    for j in range(case_num):
        if sum(train_data_matrix[i,j,:]) == 0:
            last_case_idx = j
      #      print("last_case_idx: ", j)
            break
    for k in range(last_case_idx):
        intermediate = train_data_matrix[i, k, :]
        train_data_matrix[i, k, :] = train_data_matrix[i, (k + case_num - last_case_idx), :]
        train_data_matrix[i, (k + case_num - last_case_idx), :] = intermediate

print ("after max of first case: ", max(train_data_matrix[0,0,:]))

for i in range(2910):
    # print("i is: ", i)
    last_case_idx = case_num # defaultly regard as the last one
    for j in range(case_num):
        if sum(test_data_matrix[i,j,:]) == 0:
            last_case_idx = j
       #     print("last_case_idx: ", j)
            break
    for k in range(last_case_idx):
        intermediate = test_data_matrix[i, k, :]
        test_data_matrix[i, k, :] = test_data_matrix[i, (k + case_num - last_case_idx), :]
        test_data_matrix[i, (k + case_num - last_case_idx), :] = intermediate

print ("train_data_matrix = ", train_data_matrix.shape)
print ("test_data_matrix = ", test_data_matrix.shape)

# sudo python NUHALLCOND_LSTM.py /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv
