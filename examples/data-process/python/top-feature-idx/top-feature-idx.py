# two columns: (sample num, label)
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintestvalid_data = readData(sys.argv[1])  #modify here
traintestvaliddata_matrix = np.array(traintestvalid_data[0:])[:,0:]
traintestvaliddata_matrix = traintestvaliddata_matrix.astype(np.int)
print "traintestvaliddata_matrix shape"
print traintestvaliddata_matrix.shape

file = open(sys.argv[2])
top_feature_index = np.genfromtxt(file, delimiter=",")
file.close()
print "top_feature_index.shape =\n", top_feature_index.shape

top_feature_index = top_feature_index.astype(np.int)
print "top_feature_index = \n", top_feature_index

top_k = int(sys.argv[3])
feature_dim = int (sys.argv[4])
time_dim = int (sys.argv[5])

top_k_feature_index = top_feature_index[0:top_k,0]
top_k_feature_index = top_k_feature_index - 1
#define model
topk_feature_data_matrix = np.zeros([len(traintestvaliddata_matrix[:,0]),top_k * time_dim + 1]) # last column is label
# data type correct? especially index array and label array
print "topk_feature_data_matrix.shape"
print topk_feature_data_matrix.shape
case_begin_idx = 0
copy_to_idx = 0
for i in range(time_dim):
    case_begin_idx = i * feature_dim
    for j in range(feature_dim):
        if j in top_k_feature_index:
            topk_feature_data_matrix[:,copy_to_idx] = traintestvaliddata_matrix[:,case_begin_idx + j]
            copy_to_idx = copy_to_idx + 1
print "copy_to_idx final label column = \n", copy_to_idx
print "copy from index for label column = \n", len(traintestvaliddata_matrix[0,:])
topk_feature_data_matrix[:,copy_to_idx] = traintestvaliddata_matrix[:,len(traintestvaliddata_matrix[0,:])-1]
#output
a = numpy.asarray(topk_feature_data_matrix, dtype = int)
numpy.savetxt(sys.argv[6], a, fmt = '%d', delimiter=",") #modify here
# python splittraintestvalidlabel.py 2910 4849 train_index_1.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_copy.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_train.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_test.csv NUH_DS_SOC_READMIT_DIAG_LAB_INOUT_CNN_SAMPLE_LABEL_valid.csv
