import numpy as np
import sys
from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import OneHotEncoder
import pandas

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
#traintestvalid_data = readData(sys.argv[1])  #modify here
#traintestvaliddata_matrix = np.array(traintestvalid_data[0:])[:,0:]
#print "traintestvaliddata_matrix"
#print traintestvaliddata_matrix
#print "traintestvaliddata_matrix.dtype = \n", traintestvaliddata_matrix.dtype

file = open(sys.argv[1])
data = np.genfromtxt(file, dtype = 'str', delimiter=",")
file.close()
print "data = \n",data
print "data shape = \n", data.shape
print "data.dtype = \n", data.dtype

categorical_data = np.zeros((len(data[:,0]), len(data[0,:])))
print "categorical_data.shape: \n", categorical_data.shape

categorical_data[:, 0] = data[:, 0] #SID column

for i in range(len(data[0,:])):
    if i == 0:
        continue
    a = np.array(data[:, i])
    b = pandas.get_dummies(a)
    categorical_data[:, i] = b.values.argmax(1)

print "categorical_data: \n", categorical_data
file = open(sys.argv[2], "w")
np.savetxt(file, categorical_data, "%d", ",")
file.close()
#python readfile.py balance-scale.data balance-scale.categorical.data
# python readfile.py car.data car.categorical.data
# python readfile.py house-votes-84.data house-votes-84.categorical.data
