import numpy as np
# from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_svmlight_file

def svmlightclassificationDataLoader(fileName):
    '''load classification data from fileName, transform into onehot feature representation X, and label Y
        :param
            fileName:   data file
            labelCol:   column position for label, default last column
            delimiter:  feature seperator each sample, default ','
        :returns
            X:  np feature matrix   (n_samples, n_features)
            Y:  np label matrix     (n_samples, 1)
    '''
    data = load_svmlight_file(fileName)
    X, Y = data[0], data[1]
    #print type(OneHotEncoder().fit_transform(X))
    print "svmlight X shape: ", X.shape
    print "svmlight Y shape: ", Y.shape
#    print X
    return X, Y


# X, Y = classificationDataLoader('dataset/test.data')
# print X.shape, Y.shape
# print X, Y
