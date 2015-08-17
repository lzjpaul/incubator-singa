import numpy as np
import numpy
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintest_data = readData("traintest.csv")  #modify here
traintest_label = readData("traintestlabel.csv")  #modify here
traintestdata_matrix = np.array(traintest_data[0:])[:,0:]
traintestlabel_matrix = np.array(traintest_label[0:])[:,0:]
print "traintestdata_matrix shape"
print traintestdata_matrix.shape
print "traintestlabel_matrix shape"
print traintestlabel_matrix.shape

#define model
traindata_matrix = traintestdata_matrix[0:4,:].astype(np.float) #modify here0:1018!!!!!!! less than 1018
trainlabel_matrix = traintestlabel_matrix[0:4,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "traindata_matrix.shape"
print traindata_matrix.shape
print "trainlabel_matrix.shape"
print trainlabel_matrix.shape
testdata_matrix = traintestdata_matrix[4:6:,:].astype(np.float) #modify here0:1018!!!!!!! less than 1018
testlabel_matrix = traintestlabel_matrix[4:6,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "testdata_matrix.shape"
print testdata_matrix.shape
print "testlabel_matrix.shape"
print testlabel_matrix.shape
#output
a = numpy.asarray(traindata_matrix, dtype = float)
b= numpy.asarray(trainlabel_matrix, dtype = int)
numpy.savetxt("traindata.csv", a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt("trainlabel.csv", b, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(testdata_matrix, dtype = float)
b= numpy.asarray(testlabel_matrix, dtype = int)
numpy.savetxt("testdata.csv", a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt("testlabel.csv", b, fmt = '%d', delimiter=",") #modify here
