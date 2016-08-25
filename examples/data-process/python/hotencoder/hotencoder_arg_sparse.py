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
print "train_X shape = \n", train_X.shape
#OneHotEncoder
num_feature = len(train_X[0,:])
enc = OneHotEncoder(categorical_features=[n for n in range(num_feature)],sparse=True)  #modify here!!
print "categorical_features = \n", [n for n in range(num_feature)]
enc.fit(train_X)
train_X = enc.transform(train_X)
train_X.astype(int)
print "train_X =  \n", train_X
#a = numpy.asarray(train_X, dtype = int)
#numpy.savetxt(sys.argv[2], a, fmt = '%d', delimiter=",")
#python hotencoder_arg.py NUH_DS_SOC_READMISSION_CASE_halfyear_LAB_ENGI_SUB_idxcase_demor.txt NUH_DS_SOC_READMISSION_CASE_halfyear_LAB_ENGI_SUB_idxcase_demor_onehot_arg.txt
