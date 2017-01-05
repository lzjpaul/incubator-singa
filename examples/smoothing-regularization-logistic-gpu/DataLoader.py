import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

def classificationDataLoader(fileName, labelfile, categorical_index_file, labelCol=-1, delimiter=',', sparsify=True):
    '''load classification data from fileName, transform into onehot feature representation X, and label Y
        :param
            fileName:   data file
            labelCol:   column position for label, default last column
            delimiter:  feature seperator each sample, default ','
        :returns
            X:  np feature matrix   (n_samples, n_features)
            Y:  np label matrix     (n_samples, 1)
    '''
    if labelfile is None:
        #print "label file is none"
        data = np.loadtxt(fileName, dtype='int32', delimiter=',')
        X, Y = data[:, xrange(data.shape[1]-1) if labelCol==-1 else xrange(1, data.shape[1])], data[:, labelCol]
        print "X shape in loader: ", X.shape
        #print type(OneHotEncoder().fit_transform(X))
    else:
        #print "label file is not none"
        data = np.loadtxt(fileName, dtype='int32', delimiter=',')
        label = np.loadtxt(labelfile, dtype='int32', delimiter=',')
        print "data shape in loader: ", data.shape
        X = data
        Y = label
    if categorical_index_file is None:
        return (OneHotEncoder().fit_transform(X, ), Y) if sparsify==True else (OneHotEncoder().fit_transform(X, ).toarray(), Y)
    else:
        categorical_feature_index = np.loadtxt(categorical_index_file, dtype='int32', delimiter=',')
        return (sparse.csr_matrix(OneHotEncoder(categorical_features=categorical_feature_index).fit_transform(X, ).toarray()), Y) if sparsify==True else (OneHotEncoder(categorical_features=categorical_feature_index).fit_transform(X, ).toarray(), Y)
# X, Y = classificationDataLoader('dataset/test.data')
# print X.shape, Y.shape
# print X, Y
