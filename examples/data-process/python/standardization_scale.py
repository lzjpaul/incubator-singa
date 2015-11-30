# Enrbm: divided by maximum of each column
import numpy as np
import numpy
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import argparse
import sys


#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
train_data = readData(sys.argv[2])  #modify here
train_matrix = np.array(train_data[0:])[:,0:]

fea_num = int (sys.argv[1])
#define model
print train_matrix.shape
train_X, train_y = train_matrix[:,0:fea_num].astype(np.float), train_matrix[:,fea_num].astype(np.int) #modify here0:1018!!!!!!! less than 1018
print train_X.shape
#standardization - using maximum value to normalize
train_col_max = np.zeros(len(train_X[0,:]))
for i in range (len(train_X[0,:])):
	train_col_max[i] = max(train_X[:,i])

print "train_col_max = \n", train_col_max

for i in range(len(train_X[0,:])):  # i is column number
	for j in range(len(train_X[:,0])):
		train_X[j][i] = train_X[j][i] / float(train_col_max[i])

a = numpy.asarray(train_X, dtype = float)
b= numpy.asarray(train_y, dtype = int)
numpy.savetxt(sys.argv[3], a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt(sys.argv[4], b, fmt = '%d', delimiter=",") #modify here
#python standardization.py 3 standardization_command_traintest.csv standardization_command_traintestdata_norm.csv standardization_command_traintest_label.csv
