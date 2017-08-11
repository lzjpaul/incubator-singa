# male and femal example: http://connor-johnson.com/2014/12/31/the-pearson-chi-squared-test-with-python-and-r/
import numpy as np
import sys
from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import OneHotEncoder
import pandas
import argparse
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chisquare


def cal_statistics(data):
    print "processing: ", i
    # print "data: ", data
    data_readmitted = data[data[:,-1]==1]
    data_not_readmitted = data[data[:,-1]==0]
    # print "data_readmitted: ", data_readmitted
    # print "data_not_readmitted: ", data_not_readmitted
    data_readmitted_ratio = np.sum(data_readmitted[:,0]) / len(data_readmitted[:,0])
    data_not_readmitted_ratio = np.sum(data_not_readmitted[:,0]) / len(data_not_readmitted[:,0])
    print "data_readmitted_ratio: ", data_readmitted_ratio
    print "data_not_readmitted_ratio: ", data_not_readmitted_ratio
    data_observed = data[data[:,0]==1]
    # print "data_observed: ", data_observed
    # print "observed number: ", len(data_observed[:,-1])
    print "observed readmitted: ", np.sum(data_observed[:,-1])
    print "observed not readmitted: ", (len(data_observed[:,-1]) - np.sum(data_observed[:,-1]))
    print "expected readmitted: ", (394/1755.0)*len(data_observed[:,-1])
    print "expected not readmitted: ", (1361/1755.0)*len(data_observed[:,-1])
    p_value = chisquare([np.sum(data_observed[:,-1]), len(data_observed[:,-1]) - np.sum(data_observed[:,-1])], \
                        f_exp=[(394/1755.0)*len(data_observed[:,-1]), (1361/1755.0)*len(data_observed[:,-1])])
    print "p value: ", p_value




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate p values')
    parser.add_argument('-data', type=str, help='the data path')
    parser.add_argument('-clnid', type=int, help='columnid')
    parser.add_argument('-disorcont', type=int, help='discrete or continuous variable, 0 for dicrete, 1 for continuous')
    args = parser.parse_args()

    file = open(args.data)
    data = np.genfromtxt(file, dtype = 'str', delimiter=",")
    file.close()
    # print "data = \n",data
    print "data shape = \n", data.shape
    print "data.dtype = \n", data.dtype

    data = data[:, [args.clnid, -1]]
    print "data: ", data[:10]

    if args.disorcont == 0: #discrete
    	categorical_data = np.zeros((len(data[:,0]), len(data[0,:])))
    	# print "categorical_data.shape: \n", categorical_data.shape

    	categorical_data[:, -1] = data[:, -1] #label column

    	for i in range(len(data[0,:])):
    	    if i == (len(data[0,:]) - 1):
    	        continue
    	    a = np.array(data[:, i])
    	    b = pandas.get_dummies(a)
    	    print "dummies: ", b
    	    categorical_data[:, i] = b.values.argmax(1)

    	# print "categorical_data: \n", categorical_data
    	# print "categorical_data shape: \n", categorical_data.shape

    	onehot_data = OneHotEncoder(categorical_features=[0],sparse=False).fit_transform(categorical_data.astype(int))
    	# print "onehot_data", onehot_data

    	for i in range(len(onehot_data[0,:])-1):
    	    cal_statistics(onehot_data[:, [i, -1]])


    else: #continuous variable
        data_readmitted = data[data[:,-1]=='1']
        data_not_readmitted = data[data[:,-1]=='0']
        # print "data_readmitted: ", data_readmitted[:10]
        # print "data_not_readmitted: ", data_not_readmitted[:10]
        data_readmitted = data_readmitted.astype(np.int)
        data_not_readmitted = data_not_readmitted.astype(np.int)
        readmitted_mean = np.mean(data_readmitted[:, 0])
        readmitted_std = np.std(data_readmitted[:, 0])
        not_readmitted_mean = np.mean(data_not_readmitted[:, 0])
        not_readmitted_std = np.std(data_not_readmitted[:, 0])
        print "readmitted_mean: ", readmitted_mean
        print "readmitted_std: ", readmitted_std
        print "not_readmitted_mean: ", not_readmitted_mean
        print "not_readmitted_std: ", not_readmitted_std
        data = data.astype(np.int)
        print "data[:, 0]: ", data[:, 0]
        print "data[:, 1]: ", data[:, 1]
        p_value = stats.ttest_ind(data[:, 0], data[:, 1], equal_var = False)[1]
        print "p_value: ", p_value


#python readfile.py balance-scale.data balance-scale.categorical.data
# python readfile.py car.data car.categorical.data
# python readfile.py house-votes-84.data house-votes-84.categorical.data
