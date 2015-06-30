import numpy as np
import numpy
from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
train_data = readData("uci_diabetic_data_0123_vector_lab_order_nothot.csv")
train_matrix = np.array(train_data[0:])[:,0:]

#define model
train_X = train_matrix[:,0:].astype(np.int)
#OneHotEncoder
enc = OneHotEncoder(categorical_features='all',sparse=False)
enc.fit(train_X)
train_X = enc.transform(train_X)
train_X.astype(int)
a = numpy.asarray(train_X, dtype = int)
numpy.savetxt("uci_diabetic_data_0123_vector_lab_order.csv", a, fmt = '%d', delimiter=",")
