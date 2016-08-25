import numpy as np
import numpy
from sklearn.preprocessing import OneHotEncoder
import sys

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
train_data = readData(sys.argv[1])  #modify here
train_matrix = np.array(train_data[0:])[:,0:]

#define model
train_X = train_matrix[:,0:].astype(np.int)
#OneHotEncoder
enc = OneHotEncoder(categorical_features=[0,1,2],sparse=False)  #modify here!!
enc.fit(train_X)
train_X = enc.transform(train_X)
train_X.astype(int)
a = numpy.asarray(train_X, dtype = int)
numpy.savetxt(sys.argv[2], a, fmt = '%d', delimiter=",")
