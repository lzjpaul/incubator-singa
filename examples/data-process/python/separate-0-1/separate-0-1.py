# aggregated vector, no time dimension
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
diag_train_data = readData(sys.argv[1])  #modify here
diag_train_matrix = np.array(diag_train_data[0:])[:,0:]
diag_train_0_matrix = diag_train_matrix[diag_train_matrix[:,1253]=='0'][:,0:1253]
diag_train_1_matrix = diag_train_matrix[diag_train_matrix[:,1253]=='1'][:,0:1253]
print "diag_train_matrix.shape = \n", diag_train_matrix.shape
print "diag_train_0_matrix.shape = \n", diag_train_0_matrix.shape
print "diag_train_1_matrix.shape = \n", diag_train_1_matrix.shape

lab_train_data = readData(sys.argv[2])  #modify here
lab_train_matrix = np.array(lab_train_data[0:])[:,0:]
lab_train_0_matrix = lab_train_matrix[lab_train_matrix[:,458]=='0'][:,0:458]
lab_train_1_matrix = lab_train_matrix[lab_train_matrix[:,458]=='1'][:,0:458]
print "lab_train_matrix.shape = \n", lab_train_matrix.shape
print "lab_train_0_matrix.shape = \n", lab_train_0_matrix.shape
print "lab_train_1_matrix.shape = \n", lab_train_1_matrix.shape


a = numpy.asarray(diag_train_0_matrix, dtype = int)
numpy.savetxt(sys.argv[3], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(diag_train_1_matrix, dtype = int)
numpy.savetxt(sys.argv[4], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(lab_train_0_matrix, dtype = int)
numpy.savetxt(sys.argv[5], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(lab_train_1_matrix, dtype = int)
numpy.savetxt(sys.argv[6], a, fmt = '%d', delimiter=",") #modify here

# python separate-0-1.py /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_DIAG_3_SIDGroup_longvector.txt /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LAB_SIDGroup_longvector.txt diag_train_0.csv diag_train_1.csv lab_train_0.csv lab_train_1.csv
