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
train_data = readData("standardization_example.csv")  #modify here
train_matrix = np.array(train_data[0:])[:,0:]

#define model
print train_matrix.shape
train_X, train_y = train_matrix[:,0:3].astype(np.float), train_matrix[:,3].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print train_X.shape
#standardization
scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
a = numpy.asarray(train_X, dtype = float)
b= numpy.asarray(train_y, dtype = int)
numpy.savetxt("standardization_examplefea.csv", a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt("standardization_examplelabel.csv", b, fmt = '%d', delimiter=",") #modify here
