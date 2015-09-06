# important feature
import numpy as np
import numpy.linalg as la
import numpy
#from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import argparse
import sys

# Read data in input file, which is in CSV format.
file = open(sys.argv[1])
weight_dup = np.genfromtxt(file, delimiter=",")
file.close()
print "weight_dup shape = \n", weight_dup.shape

#activation matrix
file = open(sys.argv[3])
data = np.genfromtxt(file, delimiter=",")
file.close()
print "data shape = \n", data.shape

file = open(sys.argv[6])
sample = np.genfromtxt(file, delimiter=",")
file.close()
print "sample shape = \n", sample.shape


# Arrange data into D matrix and v vector.
print "duplicated weight"
for i in range(len(weight_dup)):
    print i, weight_dup[i]

#not duplicate
weight = list(set(weight_dup))
print "not duplicated weight"
for i in range(len(weight)):
    print i, weight[i]

selected_activation_num = int (sys.argv[2])
selected_activation_index = np.zeros(selected_activation_num)
print selected_activation_index.shape

print "sorted weight\n"
print sorted(weight, reverse = True)

batchsize = len(data[:,0])
#vdim = len(data[0,:])
print "batchsize = \n", batchsize
#print "vdim = \n", vdim

data_selected_activation_col = np.empty([batchsize,selected_activation_num])

j = 0
for k in range(len(weight)):
    weight_value = sorted(weight, reverse = True)[k]
    for i in range(len(weight_dup)):
        if weight_dup[i] == weight_value and j < selected_activation_num:
            data_selected_activation_col[:,j] = data[:,i]
            j = j + 1

print "not transpose"
print data_selected_activation_col
print "transpose"
print np.matrix(data_selected_activation_col[:,:]).T
scaler = preprocessing.StandardScaler().fit(np.matrix(data_selected_activation_col[:,:]).T)
data_selected_activation_mean_dup = scaler.mean_
print "data_selected_activation transpose shape = \n", (np.matrix(data_selected_activation_col[:,:]).T).shape

print "duplicated_data_mean"
for i in range(len(data_selected_activation_mean_dup)):
    print i, data_selected_activation_mean_dup[i]

data_selected_activation_mean = list(set(data_selected_activation_mean_dup))
print "not duplicated_data_mean"
for i in range(len(data_selected_activation_mean)):
    print i, data_selected_activation_mean[i]

print "data_selected_activation_mean = \n", data_selected_activation_mean

vdim = len(sample[0,:])
selected_sample_num = int(sys.argv[4])
selected_sample = np.empty([selected_sample_num,vdim])
print sorted(data_selected_activation_mean, reverse = True)
j = 0
for k in range(len(data_selected_activation_mean)):
    mean_value = sorted(data_selected_activation_mean, reverse = True)[k]
    for i in range(len(data_selected_activation_mean_dup)):
        if data_selected_activation_mean_dup[i] == mean_value and j < selected_sample_num:
            selected_sample[j,:] = sample[i,:]
            j = j + 1
print "selected_sample = \n", selected_sample

scaler = preprocessing.StandardScaler().fit(selected_sample)
selected_sample_std_dup = scaler.std_
#print selected_sample_std
print "duplicated"
for i in range(len(selected_sample_std_dup)):
    print i, selected_sample_std_dup[i]

selected_sample_std = list(set(selected_sample_std_dup))
print "not duplicated"
for i in range(len(selected_sample_std)):
    print i, selected_sample_std[i]

selected_feature_num = int (sys.argv[5])
selected_feature_index = np.zeros(selected_feature_num)
print selected_feature_index.shape

#print selected_sample_std.shape
print sorted(selected_sample_std)

j = 0
for k in range(len(selected_sample_std)):
    feature_std_value = sorted(selected_sample_std)[k]
    for i in range(len(selected_sample_std_dup)):
	if selected_sample_std_dup[i] == feature_std_value and j < selected_feature_num:
	    selected_feature_index[j] = i
	    j = j + 1

print selected_feature_index
# Write fitting result into output file
file = open(sys.argv[7], "w")
np.savetxt(file, selected_feature_index, "%d", ",")
file.close()
#python importantfeature.py tryimportantfeatureweight.csv 3 tryimportantfeatureactivation.csv 3 4 tryimportantfeatureinputsample.csv tryimportantfeature_selected_feature_index.csv
