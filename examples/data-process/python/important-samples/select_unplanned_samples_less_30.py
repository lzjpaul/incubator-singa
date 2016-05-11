# only delete those planned and less than30 days, no change to the code
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
test_index = np.genfromtxt(file, delimiter=",")
file.close()
print "test_index.shape =\n", test_index.shape

test_index = test_index.astype(np.int)
print "test_index = \n", test_index

file = open(sys.argv[3])
planned_index = np.genfromtxt(file, delimiter=",")
file.close()
print "planned_index.shape =\n", planned_index.shape

planned_index = planned_index.astype(np.int)
print "planned_index = \n", planned_index

#read data
file = open(sys.argv[4])
train_data_matrix = np.genfromtxt(file, delimiter=",")
train_data_matrix = train_data_matrix.astype('float32')
file.close()
print ("train_data_matrix = ", train_data_matrix.shape)

#read data
file = open(sys.argv[5])
train_label_matrix = np.genfromtxt(file, delimiter=",")
train_label_matrix = train_label_matrix.astype(np.int)
file.close()
print ("train_label_matrix = ", train_label_matrix.shape)

#read data
file = open(sys.argv[6])
test_data_matrix = np.genfromtxt(file, delimiter=",")
test_data_matrix = test_data_matrix.astype('float32')
file.close()
print ("test_data_matrix = ", test_data_matrix.shape)

#read data
file = open(sys.argv[7])
test_label_matrix = np.genfromtxt(file, delimiter=",")
test_label_matrix = test_label_matrix.astype(np.int)
file.close()
print ("test_label_matrix = ", test_label_matrix.shape)

num_unplanned_train = len(train_index[:,0])
for i in range(len(train_index[:,0])):
    if train_index[i,0] in planned_index:
        num_unplanned_train = num_unplanned_train - 1
print ("len(train_index[:,0]): ", len(train_index[:,0]))
print ("num_unplanned_train: ", num_unplanned_train)

num_unplanned_test = len(test_index[:,0])
for i in range(len(test_index[:,0])):
    if test_index[i,0] in planned_index:
        num_unplanned_test = num_unplanned_test - 1
print ("len(test_index[:,0]): ", len(test_index[:,0]))
print ("num_unplanned_test: ", num_unplanned_test)

unplanned_train_data_matrix = np.zeros([num_unplanned_train,len(train_data_matrix[0,:])])
print "num_unplanned_train: ", num_unplanned_train
print "len(train_label_matrix: ", len(train_label_matrix)
unplanned_train_label_matrix = np.zeros(num_unplanned_train)
unplanned_test_data_matrix = np.zeros([num_unplanned_test,len(test_data_matrix[0,:])])
unplanned_test_label_matrix = np.zeros(num_unplanned_test)

print ("unplanned_train_data_matrix.shape: ", unplanned_train_data_matrix.shape)
print ("unplanned_train_label_matrix.shape: ", unplanned_train_label_matrix.shape)
print ("unplanned_test_data_matrix.shape: ", unplanned_test_data_matrix.shape)
print ("unplanned_test_label_matrix.shape: ", unplanned_test_label_matrix.shape)

unplanned_train_idx = 0
for i in range(len(train_index[:,0])):
    if train_index[i,0] not in planned_index:
        unplanned_train_data_matrix[unplanned_train_idx,:] = train_data_matrix[i,:]
        unplanned_train_label_matrix[unplanned_train_idx] = train_label_matrix[i]
        unplanned_train_idx = unplanned_train_idx + 1
print ("unplanned_train_idx: ", unplanned_train_idx)

unplanned_test_idx = 0
for i in range(len(test_index[:,0])):
    if test_index[i,0] not in planned_index:
        unplanned_test_data_matrix[unplanned_test_idx,:] = test_data_matrix[i,:]
        unplanned_test_label_matrix[unplanned_test_idx] = test_label_matrix[i]
        unplanned_test_idx = unplanned_test_idx + 1
print ("unplanned_test_idx: ", unplanned_test_idx)


#output
a = numpy.asarray(unplanned_train_data_matrix, dtype = float)
numpy.savetxt(sys.argv[8], a, fmt = '%6f', delimiter=",") #modify here
a = numpy.asarray(unplanned_train_label_matrix, dtype = int)
numpy.savetxt(sys.argv[9], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(unplanned_test_data_matrix, dtype = float)
numpy.savetxt(sys.argv[10], a, fmt = '%6f', delimiter=",") #modify here
a = numpy.asarray(unplanned_test_label_matrix, dtype = int)
numpy.savetxt(sys.argv[11], a, fmt = '%d', delimiter=",") #modify here
# python select_unplanned_samples.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_train.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv NUH_DS_SOC_READMISSION_planned_SID.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_train_label_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/unplanned/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_unplanned_test_label_normrange_1.csv
#('unplanned_train_idx: ', 4078)
#('unplanned_test_idx: ', 2453)
# python select_correct_0_correct_1_vague_include_incorrect.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv LSTM_1024_5_epoch/correct_label_0_index.csv LSTM_1024_5_epoch/correct_label_1_index.csv LSTM_1024_5_epoch/vague_index.csv LSTM_1024_5_epoch/incorrect_label_0_index.csv LSTM_1024_5_epoch/incorrect_label_1_index.csv LSTM_1024_5_epoch/correct_label_0_data.csv LSTM_1024_5_epoch/correct_label_1_data.csv LSTM_1024_5_epoch/vague_data.csv LSTM_1024_5_epoch/incorrect_label_0_data.csv LSTM_1024_5_epoch/incorrect_label_1_data.csv
