import numpy as np
import numpy
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintest_data = readData("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_norm_traintest.csv")  #modify here
traintest_label = readData("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_label.csv")  #modify here
traintestdata_matrix = np.array(traintest_data[0:])[:,0:]
traintestlabel_matrix = np.array(traintest_label[0:])[:,0:]
print "traintestdata_matrix shape"
print traintestdata_matrix.shape
print "traintestlabel_matrix shape"
print traintestlabel_matrix.shape

#define model
traindata_matrix = traintestdata_matrix[0:16000,:].astype(np.float) #modify here0:1018!!!!!!! less than 1018
trainlabel_matrix = traintestlabel_matrix[0:16000,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "traindata_matrix.shape"
print traindata_matrix.shape
print "trainlabel_matrix.shape"
print trainlabel_matrix.shape
testdata_matrix = traintestdata_matrix[16000:22338:,:].astype(np.float) #modify here0:1018!!!!!!! less than 1018
testlabel_matrix = traintestlabel_matrix[16000:22338,:].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print "testdata_matrix.shape"
print testdata_matrix.shape
print "testlabel_matrix.shape"
print testlabel_matrix.shape
#output
a = numpy.asarray(traindata_matrix, dtype = float)
b= numpy.asarray(trainlabel_matrix, dtype = int)
numpy.savetxt("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_norm_traindata.csv", a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_trainlabel.csv", b, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(testdata_matrix, dtype = float)
b= numpy.asarray(testlabel_matrix, dtype = int)
numpy.savetxt("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_norm_testdata.csv", a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_testlabel.csv", b, fmt = '%d', delimiter=",") #modify here
