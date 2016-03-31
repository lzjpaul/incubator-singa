import numpy as np
import numpy
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import column_or_1d
from sklearn.linear_model import LassoCV
import sys

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read all data
X_data = readData(sys.argv[1])  #modify here
X = np.array(X_data[0:])[:,0:]
X = X.astype(np.float)


y_data = readData(sys.argv[2])  #modify here
y = np.array(y_data[0:])[:,0:]
y = y.astype(np.int)
y = column_or_1d(y, warn=True)

selected_fea_num = int(sys.argv[3])

print "X.shape = \n", X.shape
print "y.shape = \n", y.shape
clf = LassoCV()
sfm = SelectFromModel(clf, threshold=0.25)
X_new = sfm.fit(X, y)
# X_new = SelectKBest(chi2, k=selected_fea_num).fit(X, y)
print "selected features = \n", X_new.get_support(indices=True)
# python feature_selection_lasso.py /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvaliddata_normrange_aggregated.csv /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvalidlabel_normrange.csv 20
