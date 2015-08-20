import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintest_data = readData(sys.argv[3])  #modify here
traintest_label = readData(sys.argv[4])  #modify here
traintestdata_matrix = np.array(traintest_data[0:])[:,0:]
traintestlabel_matrix = np.array(traintest_label[0:])[:,0:]
print "traintestdata_matrix shape"
print traintestdata_matrix.shape
print "traintestlabel_matrix shape"
print traintestlabel_matrix.shape

train_end = int (sys.argv[1])
test_end = int(sys.argv[2])
#define model
traindata_matrix = traintestdata_matrix[0:train_end,:].astype(np.float) #modify here0:1018!!!!!!! less than 1018
trainlabel_matrix = traintestlabel_matrix[0:train_end,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "traindata_matrix.shape"
print traindata_matrix.shape
print "trainlabel_matrix.shape"
print trainlabel_matrix.shape
testdata_matrix = traintestdata_matrix[train_end:test_end,:].astype(np.float) #modify here0:1018!!!!!!! less than 1018
testlabel_matrix = traintestlabel_matrix[train_end:test_end,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "testdata_matrix.shape"
print testdata_matrix.shape
print "testlabel_matrix.shape"
print testlabel_matrix.shape
#output
a = numpy.asarray(traindata_matrix, dtype = float)
b= numpy.asarray(trainlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[5], a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt(sys.argv[6], b, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(testdata_matrix, dtype = float)
b= numpy.asarray(testlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[7], a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt(sys.argv[8], b, fmt = '%d', delimiter=",") #modify here
#python splittraintest.py 18 28 test_script_traintestdata.csv test_script_traintestlabel.csv
#test_script_traindata.csv test_script_trainlabel.csv test_script_testdata.csv test_script_testlabel.csv

