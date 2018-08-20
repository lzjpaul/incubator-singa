# from sklearn.cross_validation import StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def get_data(all_feature_url, all_label_url):
    '''load data'''
    all_feature = np.genfromtxt(all_feature_url, dtype=np.float32, delimiter=',')
    all_label = np.genfromtxt(all_label_url, dtype=np.int32, delimiter=',')
    np.random.seed(10)
    # idx = np.random.permutation(all_feature.shape[0])
    # all_feature = all_feature[idx]
    # all_label = all_label[idx]
    # print "all_label.shape", all_label.shape
    return all_feature, all_label
