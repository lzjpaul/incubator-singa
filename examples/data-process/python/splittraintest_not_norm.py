import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintestlabel_data = readData(sys.argv[4])  #modify here
traintestlabeldata_matrix = np.array(traintestlabel_data[0:])[:,0:]
print "traintestdatalabel_matrix shape"
print traintestlabeldata_matrix.shape

train_end = int (sys.argv[1])
test_end = int(sys.argv[2])
feature_end = int (sys.argv[3])
#define model
testdata_matrix = traintestlabeldata_matrix[train_end:test_end,0:feature_end].astype(np.int) #modify here0:1018!!!!!!! less than 1018
testlabel_matrix = traintestlabeldata_matrix[train_end:test_end,feature_end].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "testdata_matrix.shape"
print testdata_matrix.shape
print "testlabel_matrix.shape"
print testlabel_matrix.shape
#output
a = numpy.asarray(testdata_matrix, dtype = float)
b= numpy.asarray(testlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[5], a, fmt = '%d', delimiter=",") #modify here
numpy.savetxt(sys.argv[6], b, fmt = '%d', delimiter=",") #modify here
#python splittraintest.py 18 28 test_script_traintestdata.csv test_script_traintestlabel.csv
#test_script_traindata.csv test_script_trainlabel.csv test_script_testdata.csv test_script_testlabel.csv

