# from sklearn.cross_validation import StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
from scipy.sparse import load_npz

def get_data(all_feature_url, all_label_url, all_patient_url, deployment=False):
    '''load data
    Parameters:
        all_feature_url: str, path of all_feature file
        all_label_url:  str, path of all_label file
        all_patient_url:  str, path of all_patient file
        deployment: boolean. 
            Return features and patient_id if true,
            Otherwise, return features, labels and patient_id
    
    '''
    print 'all_feature_url = ', all_feature_url
    print 'all_label_url = ', all_label_url
    print 'all_patient_url = ', all_patient_url
    print 'deployment = ', deployment
    if all_feature_url is not None and not os.path.exists(all_feature_url):
       print 'Not exist all_feature_url = ', all_feature_url
       return
    # all_feature = np.genfromtxt(all_feature_url, dtype=np.float32, delimiter=',')
    all_feature = load_npz(all_feature_url)
    all_feature = all_feature.toarray()
    
    if all_patient_url is not None and not os.path.exists(all_patient_url):
        print 'Not exist all_patient_url = ', all_patient_url
        if deployment:  # when deployment, we need patient id to insert
            return
        all_patient = []
    else:
        all_patient = np.genfromtxt(all_patient_url, dtype=str, delimiter=',')

    if deployment:
        return all_feature, all_patient

    if all_label_url is not None and not os.path.exists(all_label_url) and not deployment:
        print 'Not exist all_label_url = ', all_label_url
        return
    all_label = np.genfromtxt(all_label_url, dtype=np.int32, delimiter=',')

    return all_feature, all_label, all_patient
