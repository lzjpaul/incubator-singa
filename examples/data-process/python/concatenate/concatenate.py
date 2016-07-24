# check: some cases are all 0
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read first data matrix
first_data = readData(sys.argv[1])  #modify here
first_data_matrix = np.array(first_data[0:])[:,0:]
first_data_matrix = first_data_matrix.astype(np.float)
print "first_data_matrix shape"
print first_data_matrix.shape

#read second data matrix
second_data = readData(sys.argv[2])  #modify here
second_data_matrix = np.array(second_data[0:])[:,0:]
second_data_matrix = second_data_matrix.astype(np.float)
print "second_data_matrix shape"
print second_data_matrix.shape

concatenate_data_matrix = np.concatenate((first_data_matrix, second_data_matrix), axis=1)
print "concatenate_data_matrix shape = \n", concatenate_data_matrix.shape

#output
a = numpy.asarray(concatenate_data_matrix, dtype = float)
numpy.savetxt(sys.argv[3], a, fmt = '%6f', delimiter=",") #modify here
# python lab_laplacian.py lab_original_data.csv 2 lab_laplacian_data.csv
