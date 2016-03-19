# two columns: (sample num, label)
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintestvalid_data = readData(sys.argv[4])  #modify here
traintestvaliddata_matrix = np.array(traintestvalid_data[0:])[:,0:]
traintestvaliddata_matrix = traintestvaliddata_matrix.astype(np.int)
print "traintestvaliddata_matrix shape"
print traintestvaliddata_matrix.shape

file = open(sys.argv[3])
trainindex = np.genfromtxt(file, delimiter=",")
file.close()
print "trainindex.shape =\n", trainindex.shape

trainindex = trainindex.astype(np.int)
print "trainindex = \n", trainindex

test_end = int(sys.argv[1])
valid_end = int (sys.argv[2])
#define model
traindata_matrix = np.zeros([len(trainindex),len(traintestvaliddata_matrix[0,:])])
testvaliddata_matrix = np.zeros([(len(traintestvaliddata_matrix[:,0]) - len(trainindex)),len(traintestvaliddata_matrix[0,:])])
# data type correct? especially index array and label array
print "traindata_matrix.shape"
print traindata_matrix.shape
print "testvaliddata_matrix.shape"
print testvaliddata_matrix.shape
train_num = 0
testvalid_num = 0
for i in range(len(traintestvaliddata_matrix[:,0])):
    if i in trainindex:
        traindata_matrix[train_num,:] = traintestvaliddata_matrix[i,:]
        train_num = train_num + 1
    else:
        testvaliddata_matrix[testvalid_num,:] = traintestvaliddata_matrix[i,:]
        testvalid_num = testvalid_num + 1
print "train_num = \n", train_num
print "testvalid_num = \n", testvalid_num

testdata_matrix = testvaliddata_matrix[0:test_end,:].astype(np.int)
print "testdata_matrix.shape"
print testdata_matrix.shape

validdata_matrix = testvaliddata_matrix[test_end:valid_end,:].astype(np.int)
print "validdata_matrix.shape"
print validdata_matrix.shape

#output
a = numpy.asarray(traindata_matrix, dtype = int)
numpy.savetxt(sys.argv[5], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(testdata_matrix, dtype = int)
numpy.savetxt(sys.argv[6], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(validdata_matrix, dtype = int)
numpy.savetxt(sys.argv[7], a, fmt = '%d', delimiter=",") #modify here
# python splittraintestvalidlabel.py 2910 4849 train_index_1.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_copy.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_train.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_valid.csv
