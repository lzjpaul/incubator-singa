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
label_presigmoid = np.genfromtxt(file, delimiter=",")
file.close()
print "label_presigmoid shape = \n", label_presigmoid.shape

label = label_presigmoid[:,0].astype(np.int)
# print "label = \n", label
presigmoid_dup = label_presigmoid[:,1].astype(np.float)
print "presigmoid_dup shape = \n", presigmoid_dup.shape

#activation matrix

file = open(sys.argv[2])
sample = np.genfromtxt(file, delimiter=",")
file.close()
print "sample shape = \n", sample.shape


# Arrange data into D matrix and v vector.

#print "duplicated presigmoid"
#for i in range(len(presigmoid_dup)):
#    print i, presigmoid_dup[i]

#not duplicate
presigmoid = list(set(presigmoid_dup))
#print "not duplicated presigmoid"
#for i in range(len(presigmoid)):
#    print i, presigmoid[i]

batchsize = len(sample[:,0])
print "batchsize = \n", batchsize


vdim = len(sample[0,:])
selected_sample_num = int(sys.argv[3])
selected_sample = np.empty([selected_sample_num,vdim])
selected_sample_index = np.zeros(selected_sample_num)
# print "sorted_presigmoid = \n", sorted(presigmoid, reverse = True)
j = 0
for k in range(len(presigmoid)):
    sigmoid_value = sorted(presigmoid, reverse = True)[k]
    for i in range(len(presigmoid_dup)):
        if presigmoid_dup[i] == sigmoid_value and label[i] == 1 and j < selected_sample_num:
            selected_sample[j,:] = sample[i,:]
            selected_sample_index[j] = i
            j = j + 1

# print "selected_sample_index = \n", selected_sample_index
# print "selected_sample = \n", selected_sample

scaler = preprocessing.StandardScaler().fit(sample)
all_sample_feature_mean = scaler.mean_
scaler = preprocessing.StandardScaler().fit(selected_sample)
selected_sample_feature_mean = scaler.mean_
selected_sample_featue_max = np.zeros(vdim)
for i in range(len(selected_sample[0,:])):
    selected_sample_featue_max[i] = max(selected_sample[:,i])

# print "all_sample_feature_mean = \n", all_sample_feature_mean
# print "selected_sample_feature_mean = \n", selected_sample_feature_mean
# print "selected_sample_featue_max = \n", selected_sample_featue_max
all_and_selected_mean_difference_dup = selected_sample_feature_mean - all_sample_feature_mean
#print selected_sample_std
# print "all_and_selected_mean_difference_dup"
# for i in range(len(all_and_selected_mean_difference_dup)):
#     print i, all_and_selected_mean_difference_dup[i]

all_and_selected_mean_difference = list(set(all_and_selected_mean_difference_dup))
# print "not duplicated"
# for i in range(len(all_and_selected_mean_difference)):
#     print i, all_and_selected_mean_difference[i]

selected_feature_num = int (sys.argv[4])
selected_feature_index = np.zeros(selected_feature_num)
print "selected_feature_index.shape", selected_feature_index.shape

#print selected_sample_std.shape
# print "sorted_all_and_selected_mean_difference", sorted(all_and_selected_mean_difference, reverse = True)

j = 0
for k in range(len(all_and_selected_mean_difference)):
    difference_value = sorted(all_and_selected_mean_difference, reverse = True)[k]
    for i in range(len(all_and_selected_mean_difference_dup)):
	if all_and_selected_mean_difference_dup[i] == difference_value and j < selected_feature_num:
	    selected_feature_index[j] = i
	    j = j + 1

print selected_feature_index
# Write fitting result into output file
file = open(sys.argv[5], "w")
np.savetxt(file, selected_feature_index, "%d", ",")
file.close()

file = open(sys.argv[6], "w")
np.savetxt(file, all_and_selected_mean_difference_dup, "%5f", ",")
file.close()
#python importantfeature.py tryimportantfeaturepresigmoid.csv tryimportantfeatureinputsample.csv 3 3 tryimportantfeature_selected_feature_index.csv tryimportantfeature_feature_interestness.csv
