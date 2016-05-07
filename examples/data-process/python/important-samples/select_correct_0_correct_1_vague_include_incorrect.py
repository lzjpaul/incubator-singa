# include incorrect_0 data and incorrect_1 data
# two columns: (sample num, label)
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
test_data = readData(sys.argv[1])  # modify here
testdata_matrix = np.array(test_data[0:])[:,0:]
testdata_matrix = testdata_matrix.astype(np.int) # (sample_num, label)
print "testdata_matrix shape"
print testdata_matrix.shape

file = open(sys.argv[2])
correct_0_index = np.genfromtxt(file, delimiter=",")
file.close()
print "correct_0_index.shape =\n", correct_0_index.shape

correct_0_index = correct_0_index.astype(np.int)
print "correct_0_index = \n", correct_0_index

file = open(sys.argv[3])
correct_1_index = np.genfromtxt(file, delimiter=",")
file.close()
print "correct_1_index.shape =\n", correct_1_index.shape

correct_1_index = correct_1_index.astype(np.int)
print "correct_1_index = \n", correct_1_index

file = open(sys.argv[4])
vague_index = np.genfromtxt(file, delimiter=",")
file.close()
print "vague_index.shape =\n", vague_index.shape

vague_index = vague_index.astype(np.int)
print "vague_index = \n", vague_index

file = open(sys.argv[5])
incorrect_0_index = np.genfromtxt(file, delimiter=",")
file.close()
print "incorrect_0_index.shape =\n", incorrect_0_index.shape

incorrect_0_index = incorrect_0_index.astype(np.int)
print "incorrect_0_index = \n", incorrect_0_index

file = open(sys.argv[6])
incorrect_1_index = np.genfromtxt(file, delimiter=",")
file.close()
print "incorrect_1_index.shape =\n", incorrect_1_index.shape

incorrect_1_index = incorrect_1_index.astype(np.int)
print "incorrect_1_index = \n", incorrect_1_index


#define model
correct_0_data_matrix = np.zeros([len(correct_0_index),len(testdata_matrix[0,:])])
correct_1_data_matrix = np.zeros([len(correct_1_index),len(testdata_matrix[0,:])])
vague_data_matrix = np.zeros([len(vague_index),len(testdata_matrix[0,:])])
incorrect_0_data_matrix = np.zeros([len(incorrect_0_index),len(testdata_matrix[0,:])])
incorrect_1_data_matrix = np.zeros([len(incorrect_1_index),len(testdata_matrix[0,:])])

# data type correct? especially index array and label array
print "correct_0_data_matrix.shape"
print correct_0_data_matrix.shape
print "correct_1_data_matrix.shape"
print correct_1_data_matrix.shape
print "vague_data_matrix.shape"
print vague_data_matrix.shape
print "incorrect_0_data_matrix.shape"
print incorrect_0_data_matrix.shape
print "incorrect_1_data_matrix.shape"
print incorrect_1_data_matrix.shape

correct_0_num = 0
correct_1_num = 0
vague_num = 0
incorrect_0_num = 0
incorrect_1_num = 0

for i in range(len(testdata_matrix[:,0])):
    if i in correct_0_index:
        correct_0_data_matrix[correct_0_num,:] = testdata_matrix[i,:]
        correct_0_num = correct_0_num + 1
    elif i in correct_1_index:
        correct_1_data_matrix[correct_1_num,:] = testdata_matrix[i,:]
        correct_1_num = correct_1_num + 1
    elif i in vague_index:
        vague_data_matrix[vague_num,:] = testdata_matrix[i,:]
        vague_num = vague_num + 1
    elif i in incorrect_0_index:
        incorrect_0_data_matrix[incorrect_0_num,:] = testdata_matrix[i,:]
        incorrect_0_num = incorrect_0_num + 1
    elif i in incorrect_1_index:
        incorrect_1_data_matrix[incorrect_1_num,:] = testdata_matrix[i,:]
        incorrect_1_num = incorrect_1_num + 1

print "correct_0_num = \n", correct_0_num
print "correct_1_num = \n", correct_1_num
print "vague_num = \n", vague_num
print "incorrect_0_num = \n", incorrect_0_num
print "incorrect_1_num = \n", incorrect_1_num

#output
a = numpy.asarray(correct_0_data_matrix, dtype = int)
numpy.savetxt(sys.argv[7], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(correct_1_data_matrix, dtype = int)
numpy.savetxt(sys.argv[8], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(vague_data_matrix, dtype = int)
numpy.savetxt(sys.argv[9], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(incorrect_0_data_matrix, dtype = int)
numpy.savetxt(sys.argv[10], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(incorrect_1_data_matrix, dtype = int)
numpy.savetxt(sys.argv[11], a, fmt = '%d', delimiter=",") #modify here
# python select_correct_0_correct_1_vague_include_incorrect.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv LSTM_1024_5_epoch/correct_label_0_index.csv LSTM_1024_5_epoch/correct_label_1_index.csv LSTM_1024_5_epoch/vague_index.csv LSTM_1024_5_epoch/incorrect_label_0_index.csv LSTM_1024_5_epoch/incorrect_label_1_index.csv LSTM_1024_5_epoch/correct_label_0_data.csv LSTM_1024_5_epoch/correct_label_1_data.csv LSTM_1024_5_epoch/vague_data.csv LSTM_1024_5_epoch/incorrect_label_0_data.csv LSTM_1024_5_epoch/incorrect_label_1_data.csv
# python select_correct_0_correct_1_vague_include_incorrect.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv 1-16-1SRC-KB/correct_label_0_index.csv 1-16-1SRC-KB/correct_label_1_index.csv 1-16-1SRC-KB/vague_index.csv 1-16-1SRC-KB/incorrect_label_0_index.csv 1-16-1SRC-KB/incorrect_label_1_index.csv 1-16-1SRC-KB/correct_label_0_data.csv 1-16-1SRC-KB/correct_label_1_data.csv 1-16-1SRC-KB/vague_data.csv 1-16-1SRC-KB/incorrect_label_0_data.csv 1-16-1SRC-KB/incorrect_label_1_data.csv
# python select_correct_0_correct_1_vague.py NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv correct_label_0_index.csv correct_label_1_index.csv vague_index.csv correct_label_0_data.csv correct_label_1_data.csv vague_data.csv
