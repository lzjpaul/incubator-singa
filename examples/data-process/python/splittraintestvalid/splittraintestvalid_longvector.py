import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

test_end = int(sys.argv[1])
valid_end = int (sys.argv[2])
feature_end = int(sys.argv[3])

#read training data
traintestvalid_data_label = readData(sys.argv[5])  #modify here
traintestvalid_data_label_matrix = np.array(traintestvalid_data_label[0:])[:,0:]
traintestvaliddata_matrix = traintestvalid_data_label_matrix.astype(np.int)[:,0:feature_end]
traintestvalidlabel_matrix = traintestvalid_data_label_matrix.astype(np.int)[:,feature_end]
print "traintestvalid_data_label_matrix shape"
print traintestvalid_data_label_matrix.shape
print "traintestvaliddata_matrix shape"
print traintestvaliddata_matrix.shape
print "traintestvalidlabel_matrix shape"
print traintestvalidlabel_matrix.shape

file = open(sys.argv[4])
trainindex = np.genfromtxt(file, delimiter=",")
file.close()
print "trainindex.shape =\n", trainindex.shape

trainindex = trainindex.astype(np.int)
print "trainindex = \n", trainindex

#define model
traindata_matrix = np.zeros([len(trainindex),len(traintestvaliddata_matrix[0,:])])
trainlabel_matrix = np.zeros([len(trainindex),1])
testvaliddata_matrix = np.zeros([(len(traintestvaliddata_matrix[:,0]) - len(trainindex)),len(traintestvaliddata_matrix[0,:])])
testvalidlabel_matrix = np.zeros([(len(traintestvaliddata_matrix[:,0]) - len(trainindex)),1])
# data type correct? especially index array and label array
print "traindata_matrix.shape"
print traindata_matrix.shape
print "trainlabel_matrix.shape"
print trainlabel_matrix.shape
print "testvaliddata_matrix.shape"
print testvaliddata_matrix.shape
print "testvalidlabel_matrix.shape"
print testvalidlabel_matrix.shape
train_num = 0
testvalid_num = 0
for i in range(len(traintestvaliddata_matrix[:,0])):
    if i in trainindex:
        traindata_matrix[train_num,:] = traintestvaliddata_matrix[i,:]
        trainlabel_matrix[train_num,:] = traintestvalidlabel_matrix[i]
        train_num = train_num + 1
    else:
        testvaliddata_matrix[testvalid_num,:] = traintestvaliddata_matrix[i,:]
        testvalidlabel_matrix[testvalid_num,:] = traintestvalidlabel_matrix[i]
        testvalid_num = testvalid_num + 1
print "train_num = \n", train_num
print "testvalid_num = \n", testvalid_num

testdata_matrix = testvaliddata_matrix[0:test_end,:].astype(np.int)
testlabel_matrix = testvalidlabel_matrix[0:test_end,:].astype(np.int)
print "testdata_matrix.shape"
print testdata_matrix.shape
print "testlabel_matrix.shape"
print testlabel_matrix.shape

validdata_matrix = testvaliddata_matrix[test_end:valid_end,:].astype(np.int)
validlabel_matrix = testvalidlabel_matrix[test_end:valid_end,:].astype(np.int)
print "validdata_matrix.shape"
print validdata_matrix.shape
print "validlabel_matrix.shape"
print validlabel_matrix.shape

#output
a = numpy.asarray(traindata_matrix, dtype = int)
b= numpy.asarray(trainlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[6], a, fmt = '%d', delimiter=",") #modify here
numpy.savetxt(sys.argv[7], b, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(testdata_matrix, dtype = int)
b= numpy.asarray(testlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[8], a, fmt = '%d', delimiter=",") #modify here
numpy.savetxt(sys.argv[9], b, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(validdata_matrix, dtype = int)
b= numpy.asarray(validlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[10], a, fmt = '%d', delimiter=",") #modify here
numpy.savetxt(sys.argv[11], b, fmt = '%d', delimiter=",") #modify here
# python splittraintestvalid_longvector.py 190 315 33443
# /data/zhaojing/cnn/CMSHF/train_index/train_index_1.csv
# /data/zhaojing/cnn/CMSHF/CMS_HF_DIAG_INOUT_CNN_TIME_WEEK_longvector.txt
# /data/zhaojing/cnn/CMSHF/subsample1/CMS_HF_DIAG_INOUT_CNN_TIME_WEEK_train_data.csv
# /data/zhaojing/cnn/CMSHF/subsample1/CMS_HF_DIAG_INOUT_CNN_TIME_WEEK_train_label.csv
# /data/zhaojing/cnn/CMSHF/subsample1/CMS_HF_DIAG_INOUT_CNN_TIME_WEEK_test_data.csv
# /data/zhaojing/cnn/CMSHF/subsample1/CMS_HF_DIAG_INOUT_CNN_TIME_WEEK_test_label.csv
# /data/zhaojing/cnn/CMSHF/subsample1/CMS_HF_DIAG_INOUT_CNN_TIME_WEEK_valid_data.csv
# /data/zhaojing/cnn/CMSHF/subsample1/CMS_HF_DIAG_INOUT_CNN_TIME_WEEK_valid_label.csv
