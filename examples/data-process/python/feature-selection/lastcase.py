import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintestvalid_data = readData(sys.argv[2])  #modify here
traintestvaliddata_matrix = np.array(traintestvalid_data[0:])[:,0:]
traintestvaliddata_matrix = traintestvaliddata_matrix.astype(np.float)
print "traintestvaliddata_matrix shape"
print traintestvaliddata_matrix.shape

file = open(sys.argv[3])
lastcaseindex = np.genfromtxt(file, delimiter=",")
file.close()
print "lastcaseindex.shape =\n", lastcaseindex.shape

lastcaseindex = lastcaseindex.astype(np.int)
print "lastcaseindex = \n", lastcaseindex

feature_num = int(sys.argv[1]) # 1253
#define model
lastcasedata_matrix = np.zeros([len(lastcaseindex),feature_num])
# data type correct? especially index array and label array
print "lastcasedata_matrix.shape"
print lastcasedata_matrix.shape

for i in range(len(lastcaseindex)):
    # print "(feature_num*(lastcasedata_matrix[i,1]-1)) = ", (feature_num*(lastcasedata_matrix[i,1]-1))
    # print "(feature_num*lastcasedata_matrix[i,1]) = ", (feature_num*lastcasedata_matrix[i,1])
    # print "lastcasedata_matrix[i,:].shape: ", lastcasedata_matrix[i,:].shape
    # print "traintestvaliddata_matrix[i, (feature_num*(lastcasedata_matrix[i,1]-1)):(feature_num*lastcasedata_matrix[i,1])].shape: ", traintestvaliddata_matrix[i, (feature_num*(lastcasedata_matrix[i,1]-1)):(feature_num*lastcasedata_matrix[i,1])].shape
    lastcasedata_matrix[i,:] = traintestvaliddata_matrix[i, (feature_num*(lastcaseindex[i,1]-1)):(feature_num*lastcaseindex[i,1])]
#output
a = numpy.asarray(lastcasedata_matrix, dtype = float)
numpy.savetxt(sys.argv[4], a, fmt = '%6f', delimiter=",") #modify here
#python splittraintest.py 18 28 test_script_traintestdata.csv test_script_traintestlabel.csv
# python lastcase.py 1253 /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvaliddata_normrange.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_DIAG_1src_CaseNum.csv /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvaliddata_normrange_lastcase.csv
