import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
traintestvalid_data = readData(sys.argv[4])  #modify here
traintestvalid_label = readData(sys.argv[5])  #modify here
traintestvaliddata_matrix = np.array(traintestvalid_data[0:])[:,0:]
traintestvalidlabel_matrix = np.array(traintestvalid_label[0:])[:,0:]
traintestvaliddata_matrix = traintestvaliddata_matrix.astype(np.float)
traintestvalidlabel_matrix = traintestvalidlabel_matrix.astype(np.int)
print "traintestvaliddata_matrix shape"
print traintestvaliddata_matrix.shape
print "traintestvalidlabel_matrix shape"
print traintestvalidlabel_matrix.shape

file = open(sys.argv[3])
trainindex = np.genfromtxt(file, delimiter=",")
file.close()
print "trainindex.shape =\n", trainindex.shape

trainindex = trainindex.astype(np.int)
print "trainindex = \n", trainindex

test_end = int(sys.argv[1])
valid_end = int (sys.argv[2])
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
        trainlabel_matrix[train_num,:] = traintestvalidlabel_matrix[i,:]
        train_num = train_num + 1
    else:
        testvaliddata_matrix[testvalid_num,:] = traintestvaliddata_matrix[i,:]
        testvalidlabel_matrix[testvalid_num,:] = traintestvalidlabel_matrix[i,:]
        testvalid_num = testvalid_num + 1
print "train_num = \n", train_num
print "testvalid_num = \n", testvalid_num

testdata_matrix = testvaliddata_matrix[0:test_end,:].astype(np.float)
testlabel_matrix = testvalidlabel_matrix[0:test_end,:].astype(np.int)
print "testdata_matrix.shape"
print testdata_matrix.shape
print "testlabel_matrix.shape"
print testlabel_matrix.shape

validdata_matrix = testvaliddata_matrix[test_end:valid_end,:].astype(np.float)
validlabel_matrix = testvalidlabel_matrix[test_end:valid_end,:].astype(np.int)
print "validdata_matrix.shape"
print validdata_matrix.shape
print "validlabel_matrix.shape"
print validlabel_matrix.shape

#output
a = numpy.asarray(traindata_matrix, dtype = float)
b= numpy.asarray(trainlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[6], a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt(sys.argv[7], b, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(testdata_matrix, dtype = float)
b= numpy.asarray(testlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[8], a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt(sys.argv[9], b, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(validdata_matrix, dtype = float)
b= numpy.asarray(validlabel_matrix, dtype = int)
numpy.savetxt(sys.argv[10], a, fmt = '%6f', delimiter=",") #modify here
numpy.savetxt(sys.argv[11], b, fmt = '%d', delimiter=",") #modify here
#python splittraintest.py 18 28 test_script_traintestdata.csv test_script_traintestlabel.csv
#test_script_traindata.csv test_script_trainlabel.csv test_script_testdata.csv test_script_testlabel.csv
# python splittraintestvalid.py 3000 5000 /data/zhaojing/marble/mlp-samples/subsample1/Marble_split_trainindex_1.csv /data/zhaojing/marble/mlp-samples/Marble_08_09_Car_Cla_UMLS_traintest_data_norm.csv /data/zhaojing/marble/mlp-samples/Marble_08_09_Car_Cla_UMLS_traintest_label.csv /data/zhaojing/marble/mlp-samples/subsample1/Marble_08_09_Car_Cla_UMLS_train_data_norm_1.csv /data/zhaojing/marble/mlp-samples/subsample1/Marble_08_09_Car_Cla_UMLS_train_label_1.csv /data/zhaojing/marble/mlp-samples/subsample1/Marble_08_09_Car_Cla_UMLS_test_data_norm_1.csv /data/zhaojing/marble/mlp-samples/subsample1/Marble_08_09_Car_Cla_UMLS_test_label_1.csv /data/zhaojing/marble/mlp-samples/subsample1/Marble_08_09_Car_Cla_UMLS_valid_data_norm_1.csv /data/zhaojing/marble/mlp-samples/subsample1/Marble_08_09_Car_Cla_UMLS_valid_label_1.csv
