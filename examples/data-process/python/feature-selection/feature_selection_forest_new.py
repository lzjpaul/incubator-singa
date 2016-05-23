# new: can sort the indices......
import numpy as np
import numpy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import column_or_1d
from sklearn.linear_model import LassoCV
from sklearn.svm import LinearSVC
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

n_jobs = 1
# is it correct? need to selectfrommodel??
print "X.shape = \n", X.shape
print "y.shape = \n", y.shape
forest = ExtraTreesClassifier(n_estimators=1000, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
# X_new = SelectKBest(chi2, k=selected_fea_num).fit(X, y)
print "importances = \n", importances
print "indices = \n", np.argsort(importances)[::-1][:selected_fea_num]
# print "important indices = \n", (-np.array(importances)).argsort[:20]
# python feature_selection_forest_new.py /ssd/zhaojing/cnn/NUHALLCOND/VISIT_DIAG_aggcnt_1/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_train_data_normrange_1.csv /ssd/zhaojing/cnn/NUHALLCOND/VISIT_DIAG_aggcnt_1/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_train_label_normrange_1.csv 2
# python feature_selection_forest.py /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvaliddata_normrange_aggregated.csv /ssd/zhaojing/cnn/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_traintestvalidlabel_normrange.csv 20
