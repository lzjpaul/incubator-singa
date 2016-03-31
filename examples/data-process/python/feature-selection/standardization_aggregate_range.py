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
train_data = readData(sys.argv[3])  #modify here
train_matrix = np.array(train_data[0:])[:,0:]
train_matrix = train_matrix.astype(np.float)

fea_num = int (sys.argv[1]) # 1253
time_dimension = int (sys.argv[2]) # 12

#9699 samples * 1253 features
aggregated_data_matrix = np.zeros([len(train_matrix[:,0]), fea_num])
print "aggregated_data_matrix.shape = ", aggregated_data_matrix.shape

#for i in range(fea_num):
#    for j in range(time_dimension):
#        for k in range(len(train_matrix[:, 0])):
#            aggregated_data_matrix[k,i] = aggregated_data_matrix[k,i] + train_matrix[k, (i + j * fea_num)]

for i in range(fea_num):
    for j in range(time_dimension):
        aggregated_data_matrix[:,i] = aggregated_data_matrix[:,i] + train_matrix[:, (i + j * fea_num)]

#define model
aggregated_train_X = aggregated_data_matrix[:,0:fea_num].astype(np.float) #modify here0:1018!!!!!!! less than 1018
print "aggregated_train_X.shape = ", aggregated_train_X.shape
#standardization
a = numpy.asarray(aggregated_train_X, dtype = float)
numpy.savetxt(sys.argv[4], a, fmt = '%2f', delimiter=",") #modify here
min_max_scaler = preprocessing.MinMaxScaler()
aggregated_train_X = min_max_scaler.fit_transform(aggregated_train_X)
a = numpy.asarray(aggregated_train_X, dtype = float)
numpy.savetxt(sys.argv[5], a, fmt = '%6f', delimiter=",") #modify here
# python standardization_aggregate_range.py 1253 12 /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_longvector.txt /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvaliddata_aggregated.csv /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvaliddata_normrange_aggregated.csv
