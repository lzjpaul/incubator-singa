import itertools
import numpy as np
from sklearn import metrics

from sptensor import sptensor


def get_model_AUC(model, featMat, Y, trainIdx, testIdx):
    """
    Get the AUC for the model by fitting the training data and the test data

    Parameters
    ------------
    model: scikit-learn classifier (e.g. logistic)
           model object (needs to have a fit and predict_proba function)
    featMat: 2-d np.ndarray
             the feature matrix (sample x feature)
    Y : the class type for each sample of the feature matrix
    trainIdx : the sample indices are in the training set
    testIdx : the sample indices associated with the test set

    Returns
    -------
    M : np.ndarray
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of ``U``

    """
    trainY = Y[trainIdx]
    model.fit(featMat[trainIdx, :], trainY)
    modelPred = model.predict_proba(featMat[testIdx, :])
    fpr, tpr, thresholds = metrics.roc_curve(Y[testIdx], modelPred[:, 1],
                                             pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, modelPred


def create_raw_features(X):
    mode2Offset = X.shape[1]
    rawFeat = np.zeros((X.shape[0], mode2Offset + X.shape[2]))
    for k in range(X.subs.shape[0]):
        sub = X.subs[k, :]
        rawFeat[sub[0], sub[1]] = rawFeat[sub[0], sub[1]] + X.vals[k, 0]
        new_col = mode2Offset + sub[2]
        rawFeat[sub[0], new_col] = rawFeat[sub[0],
                                           new_col] + X.vals[k, 0]
    return rawFeat


def rebase_indices(ids, subs):
    """
    Re-index according to the ordered array that specifies the new indices

    Parameters
    ------------
    ids : ordered array that embeds the location
    subs : the locations that need to be 'reindexed' according to the ids
    """
    idMap = dict(itertools.izip(ids, range(len(ids))))
    for k in range(subs.shape[0]):
        id = subs[k, 0]
        subs[k, 0] = idMap[id]
    return subs


def subset_sptensor(origTensor, subsetIds, subsetShape):
    """
    Get a subset of the tensor specified by the subsetIds

    Parameters
    ------------
    X : the original tensor
    subsetIds : a list of indices
    subsetShape : the shape of the new tensor

    Output
    -----------
    subsetX : the tensor with the indices rebased
    """
    subsetIdx = np.in1d(origTensor.subs[:, 0].ravel(), subsetIds)
    subsIdx = np.where(subsetIdx)[0]
    subsetSubs = origTensor.subs[subsIdx, :]
    subsetVals = origTensor.vals[subsIdx]
    # reindex the 0th mode
    subsetSubs = rebase_indices(subsetIds, subsetSubs)
    return sptensor(subsetSubs, subsetVals, subsetShape)
