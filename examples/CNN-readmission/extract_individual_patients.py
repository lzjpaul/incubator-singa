from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from argparse import ArgumentParser

def main():
    '''Command line options'''
    # Setup argument parser
    parser = ArgumentParser(description="Extract Test sample index")
    parser.add_argument('-allfeatureurl', type=str, help='the test feature path')
    parser.add_argument('-alllabelurl', type=str, help='the test label path')
        
    # Process arguments
    args = parser.parse_args()
        
    '''load data'''
    all_feature = np.genfromtxt(args.allfeatureurl, dtype=np.float32, delimiter=',')
    all_label = np.genfromtxt(args.alllabelurl, dtype=np.int32, delimiter=',')
    np.random.seed(10)
    idx = np.random.permutation(all_feature.shape[0])
    all_label_origin = np.copy(all_label)
    all_feature = all_feature[idx]
    all_label = all_label[idx]
    n_folds = 5
    for i, (train_index, test_index) in enumerate(StratifiedKFold(all_label.reshape(all_label.shape[0]), n_folds=n_folds)):
        train_feature, train_label, test_feature, test_label = all_feature[train_index], all_label[train_index], all_feature[test_index], all_label[test_index]
        test_idx = idx[test_index]
        print test_idx
        np.savetxt('nuh_fa_readmission_case_demor_inpa_fold_0_test_index.csv', test_idx.reshape((-1, 1)), fmt = '%d', delimiter=",") #modify here
        print all_label_origin[test_idx]
        if i == 0:
            print "fold: ", i
            break
     


if __name__ == '__main__':
    main()
