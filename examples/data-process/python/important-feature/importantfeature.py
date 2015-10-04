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
        if j >= selected_sample_num:
            break
        else:
            if presigmoid_dup[i] == sigmoid_value and label[i] == 1 and j < selected_sample_num:
                selected_sample[j,:] = sample[i,:]
                selected_sample_index[j] = i
                j = j + 1

print "selected_sample_index = \n", selected_sample_index
# print "selected_sample = \n", selected_sample
# print "diabetes selected_sample = \n", selected_sample[:,226]

scaler = preprocessing.StandardScaler().fit(sample)
all_sample_feature_mean = scaler.mean_
all_sample_feature_max = np.zeros(vdim)
all_sample_feature_min = np.zeros(vdim)
for i in range(vdim):
    all_sample_feature_max[i] = max(sample[:,i])
    all_sample_feature_min[i] = min(sample[:,i])

scaler = preprocessing.StandardScaler().fit(selected_sample)
selected_sample_feature_mean = scaler.mean_
selected_sample_feature_max = np.zeros(vdim)
selected_sample_feature_min = np.zeros(vdim)
for i in range(len(selected_sample[0,:])):
    selected_sample_feature_max[i] = max(selected_sample[:,i])
    selected_sample_feature_min[i] = min(selected_sample[:,i])

# print "diabetes selected_sample mean = \n", selected_sample_feature_mean[226]
# print "diabetes selected_sample max = \n", selected_sample_featue_max[226]
for i in range(len(all_sample_feature_mean)):
    if all_sample_feature_mean[i] < 1.0000000e-10:
        all_sample_feature_mean[i] = 100000000
print "all_sample_feature_mean = \n", all_sample_feature_mean
print "selected_sample_feature_mean = \n", selected_sample_feature_mean
# impossible value because it never appears in this sample
# print "selected_sample_featue_max = \n", selected_sample_featue_max
all_and_selected_mean_difference_dup = [float(c)/t for c,t in zip((selected_sample_feature_mean - all_sample_feature_mean), all_sample_feature_mean)]
print "all_and_selected_mean_difference_dup = \n", all_and_selected_mean_difference_dup
#print selected_sample_std
# print "all_and_selected_mean_difference_dup"
# for i in range(len(all_and_selected_mean_difference_dup)):
#     print i, all_and_selected_mean_difference_dup[i]

all_and_selected_mean_difference = list(set(all_and_selected_mean_difference_dup))
print "not duplicated_difference = \n", all_and_selected_mean_difference
# for i in range(len(all_and_selected_mean_difference)):
#     print i, all_and_selected_mean_difference[i]

selected_feature_num = int (sys.argv[4])
selected_feature_index = np.zeros(selected_feature_num)
print "selected_feature_index.shape = \n", selected_feature_index.shape

#print selected_sample_std.shape
print "sorted_all_and_selected_mean_difference", sorted(all_and_selected_mean_difference, reverse = True)

j = 0
for k in range(len(all_and_selected_mean_difference)):
    difference_value = sorted(all_and_selected_mean_difference, reverse = True)[k]
    for i in range(len(all_and_selected_mean_difference_dup)):
        if j >= selected_feature_num:
            break
        else:
	    if all_and_selected_mean_difference_dup[i] == difference_value and j < selected_feature_num:
                print "difference_value = ", difference_value
	        selected_feature_index[j] = i
	        j = j + 1
selected_feature_index = selected_feature_index.astype(np.int)
print "selected_feature_index = \n", selected_feature_index

selected_feature_all_sample_mean = np.zeros(selected_feature_num)
selected_feature_all_sample_max = np.zeros(selected_feature_num)
selected_feature_all_sample_min = np.zeros(selected_feature_num)
selected_feature_selected_sample_mean = np.zeros(selected_feature_num)
selected_feature_selected_sample_max = np.zeros(selected_feature_num)
selected_feature_selected_sample_min = np.zeros(selected_feature_num)
selected_feature_difference = np.zeros(selected_feature_num)
for i in range(len(selected_feature_index)):
    selected_feature_all_sample_mean[i] = all_sample_feature_mean[selected_feature_index[i]]
    selected_feature_all_sample_max[i] = all_sample_feature_max[selected_feature_index[i]]
    selected_feature_all_sample_min[i] = all_sample_feature_min[selected_feature_index[i]]
    selected_feature_selected_sample_mean[i] = selected_sample_feature_mean[selected_feature_index[i]]
    selected_feature_selected_sample_max[i] = selected_sample_feature_max[selected_feature_index[i]]
    selected_feature_selected_sample_min[i] = selected_sample_feature_min[selected_feature_index[i]]
    selected_feature_difference[i] = all_and_selected_mean_difference_dup[selected_feature_index[i]]
# Write fitting result into output file
file = open(sys.argv[5], "w")
np.savetxt(file, selected_feature_index, "%d", ",")
file.close()

print "selected_feature_difference = \n", selected_feature_difference
file = open(sys.argv[6], "w")
np.savetxt(file, selected_feature_difference, "%5f", ",")
file.close()

print "selected_feature_all_sample_mean = \n", selected_feature_all_sample_mean
file = open(sys.argv[7], "w")
np.savetxt(file, selected_feature_all_sample_mean, "%f", ",")
file.close()

print "selected_feature_all_sample_max = \n", selected_feature_all_sample_max
print "selected_feature_all_sample_min = \n", selected_feature_all_sample_min

print "selected_feature_selected_sample_mean = \n", selected_feature_selected_sample_mean
file = open(sys.argv[8], "w")
np.savetxt(file, selected_feature_selected_sample_mean, "%5f", ",")
file.close()

print "selected_feature_selected_sample_max = \n", selected_feature_selected_sample_max
print "selected_feature_selected_sample_min = \n", selected_feature_selected_sample_min
# top 10 features that have higher proportion of mean
#python importantfeature.py tryimportantfeaturepresigmoid.csv tryimportantfeatureinputsample.csv 3 3 tryimportantfeature_selected_feature_index.csv tryimportantfeature_feature_interestness.csv
# python importantfeature.py MarbleResult/01-500-2.csv /data/zhaojing/marble/importantfeature/Marble_08_09_Car_Cla_UMLS_important_features_test_data.csv 100 10 MarbleResult/Marble_08_09_Car_Cla_UMLS_important_features_feature_index.csv MarbleResult/Marble_08_09_Car_Cla_UMLS_important_features_intereness.csv MarbleResult/Marble_08_09_Car_Cla_UMLS_important_features_all_sample_mean.csv MarbleResult/Marble_08_09_Car_Cla_UMLS_important_features_selected_sample_mean.csv
