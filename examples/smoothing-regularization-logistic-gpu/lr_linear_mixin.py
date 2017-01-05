#logistic output
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
#Mixin for logisitc linear classifiers.
#    Handles prediction for sparse and dense X.
#    """

class LogisticLinearClassifierMixin(LinearClassifierMixin):

    def decision_function(self, X):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        print "in LogisticLinearClassifierMixin 12121212"
        print "in LogisticLinearClassifierMixin  X norm: ", np.linalg.norm(X)
        print "in LogisticLinearClassifierMixin  X shape: ", X.shape

        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted"
                                 "yet" % {'name': type(self).__name__})

        X = check_array(X, accept_sparse='csr')

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        print "in lr_linear_mixin.py Logistic dot 11-11-11"
        print "in lr_linear_mixin.py self.intercept_ shape: ", self.intercept_.shape
        print "in lr_linear_mixin.py self.intercept_ norm: ", np.linalg.norm(self.intercept_)
        print "in lr_linear_mixin.py self.coef_: ", self.coef_
        scores = safe_sparse_dot(X, self.coef_.T,
                                 dense_output=True) + self.intercept_
        scores = 1. / (1. + np.exp(-scores))
        print "in lr_linear_mixin.py decision function scores norm: ", np.linalg.norm(scores)
        print "in lr_linear_mixin.py decision function max scores: ", max(scores)
        print "in lr_linear_mixin.py decision function min scores: ", min(scores)
        return scores.ravel() if scores.shape[1] == 1 else scores
