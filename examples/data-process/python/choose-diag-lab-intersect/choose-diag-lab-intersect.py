# include incorrect_0 data and incorrect_1 data
# two columns: (sample num, label)
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

file = open(sys.argv[1])
train_index = np.genfromtxt(file, delimiter=",")
file.close()
print "train_index.shape =\n", train_index.shape

train_index = train_index.astype(np.int)
print "train_index = \n", train_index

file = open(sys.argv[2])
intersect_index = np.genfromtxt(file, delimiter=",")
file.close()
print "intersect_index.shape =\n", intersect_index.shape

intersect_index = intersect_index.astype(np.int)
print "intersect_index = \n", intersect_index

#read data
file = open(sys.argv[3])
train_data_matrix = np.genfromtxt(file, delimiter=",")
train_data_matrix = train_data_matrix.astype('float32')
file.close()
print ("train_data_matrix = ", train_data_matrix.shape)

if len(train_index) != len(train_data_matrix[:,0]):
    print "error!! len(train_index) != len(train_data_matrix[:,0])"
#read data
#file = open(sys.argv[5])
#train_label_matrix = np.genfromtxt(file, delimiter=",")
#train_label_matrix = train_label_matrix.astype(np.int)
#file.close()
#print ("train_label_matrix = ", train_label_matrix.shape)

intersect_train_data_matrix = np.zeros([len(intersect_index),len(train_data_matrix[0,:])])
print "num_intersect_train: ", len(intersect_index)
print "intersect_train_data_matrix.shape: ", intersect_train_data_matrix.shape

intersect_train_idx = 0
print "len(train_index[:,0]): ", len(train_index)
for i in range(len(train_index)):
    if train_index[i] in intersect_index:
        intersect_train_data_matrix[intersect_train_idx,:] = train_data_matrix[i,:]
        intersect_train_idx = intersect_train_idx + 1
print "intersect_train_idx: ", intersect_train_idx
print "intersect_train_data_matrix: ", intersect_train_data_matrix.shape

#output
a = numpy.asarray(intersect_train_data_matrix, dtype = float)
numpy.savetxt(sys.argv[4], a, fmt = '%6f', delimiter=",") #modify here
# python choose-diag-lab-intersect.py NUHDSSOCREADMI_DIAGLAB_ENGI_SUB_IDX_LAB_SID.csv NUHDSSOCREADMI_DIAGLAB_ENGI_SUB_IDX_DIAG_LAB_INTERSECT_SID.csv /ssd/zhaojing/lab-laplacian/idx-case/all-case-idx-case/NUH_DS_SOC_READMISSION_CASE_LAB_ENGI_SUB_idxcase_demor_onehot.txt /ssd/zhaojing/lab-laplacian/idx-case/all-case-idx-case/intersect/NUH_DS_SOC_READMISSION_CASE_LAB_ENGI_SUB_idxcase_demor_onehot_INTERSECT.txt
# python select_unplanned_samples.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_train.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv NUH_DS_SOC_READMISSION_planned_SID.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_label_normrange_1.csv
#('unplanned_train_idx: ', 4078)
#('unplanned_test_idx: ', 2453)
# python select_correct_0_correct_1_vague_include_incorrect.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv LSTM_1024_5_epoch/correct_label_0_index.csv LSTM_1024_5_epoch/correct_label_1_index.csv LSTM_1024_5_epoch/vague_index.csv LSTM_1024_5_epoch/incorrect_label_0_index.csv LSTM_1024_5_epoch/incorrect_label_1_index.csv LSTM_1024_5_epoch/correct_label_0_data.csv LSTM_1024_5_epoch/correct_label_1_data.csv LSTM_1024_5_epoch/vague_data.csv LSTM_1024_5_epoch/incorrect_label_0_data.csv LSTM_1024_5_epoch/incorrect_label_1_data.csv
