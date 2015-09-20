import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintest_label = readData(sys.argv[3])  #modify here
traintestlabel_matrix = np.array(traintest_label[0:])[:,0:]
print "traintestlabel_matrix shape"
print traintestlabel_matrix.shape

train_end = int (sys.argv[1])
test_end = int(sys.argv[2])
#define model
trainlabel_matrix = traintestlabel_matrix[0:train_end,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "trainlabel_matrix.shape"
print trainlabel_matrix.shape
testlabel_matrix = traintestlabel_matrix[train_end:test_end,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "testlabel_matrix.shape"
print testlabel_matrix.shape
#output
b= numpy.asarray(trainlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[4], b, fmt = '%d', delimiter=",") #modify here
b= numpy.asarray(testlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[5], b, fmt = '%d', delimiter=",") #modify here
#python splittraintestlabel.py 18 28 test_script_traintestlabel.csv test_script_trainlabel.csv test_script_testlabel.csv

