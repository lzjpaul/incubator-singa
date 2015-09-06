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
data = np.genfromtxt(file, delimiter=",")
file.close()
print "data shape = \n", data.shape

file = open(sys.argv[2])
index = np.genfromtxt(file, delimiter=",")
file.close()
print "data index = \n", index

# Arrange data into D matrix and v vector.
select_sample_n = len(index)
col = len(data[0, :])
selected_sample = np.matrix(np.empty([select_sample_n,col]))

for i in range(len(index)):
	selected_sample[i,:] = data[index[i],:]

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

selected_feature_num = int (sys.argv[4])
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
file = open(sys.argv[3], "w")
np.savetxt(file, selected_feature_index, "%d", ",")
file.close()
#python importantfeature.py importantfeatureinputsample.csv importantfeaturesampleindex.csv importantfeatureindex.txt 2
