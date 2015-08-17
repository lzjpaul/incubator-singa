import numpy as np
from sklearn.metrics import roc_auc_score

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
train_data = readData("aurocinput.csv")  #modify here
train_matrix = np.array(train_data[0:])[:,0:]

#define model
print train_matrix.shape
y_true, y_scores = train_matrix[:,0:1].astype(np.int), train_matrix[:,1].astype(np.float) #modify here0:1018!!!!!!! less than 1018
print y_true.shape
#auroc score
print roc_auc_score(y_true, y_scores)
