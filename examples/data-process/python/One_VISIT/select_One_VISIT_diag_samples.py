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
diag_index = np.genfromtxt(file, delimiter=",")
file.close()
print "diag_index.shape =\n", diag_index.shape

diag_index = diag_index.astype(np.int)
print "diag_index = \n", diag_index

#read data
file = open(sys.argv[2])
train_data_matrix = np.genfromtxt(file, delimiter=",")
train_data_matrix = train_data_matrix.astype('float32')
file.close()
print ("train_data_matrix = ", train_data_matrix.shape)

#read data
file = open(sys.argv[3])
train_label_matrix = np.genfromtxt(file, delimiter=",")
train_label_matrix = train_label_matrix.astype(np.int)
file.close()
print ("train_label_matrix = ", train_label_matrix.shape)

diag_train_data_matrix = np.zeros([len(diag_index),len(train_data_matrix[0,:])])
print "diag_train_data_matrix.shape: ", diag_train_data_matrix.shape
print "len(train_label_matrix: ", len(train_label_matrix)
diag_train_label_matrix = np.zeros(len(diag_index))
print ("diag_train_label_matrix.shape: ", diag_train_label_matrix.shape)

diag_train_idx = 0
for i in range(len(train_data_matrix[:,0])):
    if (i+1) in diag_index:
        diag_train_data_matrix[diag_train_idx,:] = train_data_matrix[i,:]
        diag_train_label_matrix[diag_train_idx] = train_label_matrix[i]
        diag_train_idx = diag_train_idx + 1
print ("diag_train_idx: ", diag_train_idx)

#output
a = numpy.asarray(diag_train_data_matrix, dtype = float)
numpy.savetxt(sys.argv[4], a, fmt = '%6f', delimiter=",") #modify here
a = numpy.asarray(diag_train_label_matrix, dtype = int)
numpy.savetxt(sys.argv[5], a, fmt = '%d', delimiter=",") #modify here
# python select_One_VISIT_diag_samples.py /data/zhaojing/rnn/CMS_HF_one_VISIT/CMS_HF_one_VISIT_DS_DIAG_SID_LIST.csv /data/zhaojing/rnn/CMS_HF_one_VISIT/CMS_HF_one_VISIT_DS_DIAG_features.txt /data/zhaojing/rnn/CMS_HF_one_VISIT/CMS_HF_READMISSION_CASE_label.csv /data/zhaojing/rnn/CMS_HF_one_VISIT/CMS_HF_one_VISIT_DS_DIAG_intersect_features.txt /data/zhaojing/rnn/CMS_HF_one_VISIT/CMS_HF_READMISSION_CASE_intersect_label.csv
# python select_unplanned_samples.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_train.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv NUH_DS_SOC_READMISSION_planned_SID.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_label_normrange_1.csv
#('unplanned_train_idx: ', 4078)
#('unplanned_test_idx: ', 2453)
# python select_correct_0_correct_1_vague_include_incorrect.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv LSTM_1024_5_epoch/correct_label_0_index.csv LSTM_1024_5_epoch/correct_label_1_index.csv LSTM_1024_5_epoch/vague_index.csv LSTM_1024_5_epoch/incorrect_label_0_index.csv LSTM_1024_5_epoch/incorrect_label_1_index.csv LSTM_1024_5_epoch/correct_label_0_data.csv LSTM_1024_5_epoch/correct_label_1_data.csv LSTM_1024_5_epoch/vague_data.csv LSTM_1024_5_epoch/incorrect_label_0_data.csv LSTM_1024_5_epoch/incorrect_label_1_data.csv
