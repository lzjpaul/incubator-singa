# no changes to the code, this one only excludes planned and less than 30 days
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
test_index = np.genfromtxt(file, delimiter=",")
file.close()
print "test_index.shape =\n", test_index.shape

test_index = test_index.astype(np.int)
print "test_index = \n", test_index

file = open(sys.argv[2])
planned_index = np.genfromtxt(file, delimiter=",")
file.close()
print "planned_index.shape =\n", planned_index.shape

planned_index = planned_index.astype(np.int)
print "planned_index = \n", planned_index

#read data
file = open(sys.argv[3])
test_prediction_matrix = np.genfromtxt(file, delimiter=",")
test_prediction_matrix = test_prediction_matrix.astype('float32')
file.close()
print ("test_prediction_matrix = ", test_prediction_matrix.shape)

#read data
file = open(sys.argv[4])
test_label_matrix = np.genfromtxt(file, delimiter=",")
test_label_matrix = test_label_matrix.astype(np.int)
file.close()
print ("test_label_matrix = ", test_label_matrix.shape)


planned_test_idx = 0
planned_less_than_05 = 0
for i in range(len(test_index[:,0])):
    if test_index[i,0] in planned_index:
        planned_test_idx = planned_test_idx + 1
        if test_prediction_matrix[i] < 0.5:
            planned_less_than_05 = planned_less_than_05 + 1  
print ("planned_test_idx: ", planned_test_idx)
print ("planned_less_than_05: ", planned_less_than_05)

# python select_unplanned_samples.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_train.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv NUH_DS_SOC_READMISSION_planned_SID.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_label_normrange_1.csv
#('unplanned_train_idx: ', 4078)
#('unplanned_test_idx: ', 2453)
# python select_correct_0_correct_1_vague_include_incorrect.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv LSTM_1024_5_epoch/correct_label_0_index.csv LSTM_1024_5_epoch/correct_label_1_index.csv LSTM_1024_5_epoch/vague_index.csv LSTM_1024_5_epoch/incorrect_label_0_index.csv LSTM_1024_5_epoch/incorrect_label_1_index.csv LSTM_1024_5_epoch/correct_label_0_data.csv LSTM_1024_5_epoch/correct_label_1_data.csv LSTM_1024_5_epoch/vague_data.csv LSTM_1024_5_epoch/incorrect_label_0_data.csv LSTM_1024_5_epoch/incorrect_label_1_data.csv
