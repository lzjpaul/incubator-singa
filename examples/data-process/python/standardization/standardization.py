import numpy as np
import numpy
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import argparse


#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
train_data = readData("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down.txt")  #modify here
train_matrix = np.array(train_data[0:])[:,0:]

#define model
print train_matrix.shape
train_X, train_y = train_matrix[:,0:692].astype(np.float), train_matrix[:,692].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print train_X.shape
#standardization
scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
a = numpy.asarray(train_X, dtype = float)
b= numpy.asarray(train_y, dtype = int)
numpy.savetxt("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_norm.csv", a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt("/data/zhaojing/uci-regularization/uci_diabetic_data_regulariz_longvector_down_label.csv", b, fmt = '%d', delimiter=",") #modify here
