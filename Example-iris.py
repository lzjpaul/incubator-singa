# n_folds = 5
from huber_svm import HuberSVC

import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.preprocessing import scale

data = load_iris()

X = scale(data['data'])
y = data['target']

idx = np.random.permutation(X.shape[0])
X = X[idx]
y = y[idx]

lasso = OneVsRestClassifier(Lasso())
param_lasso = {'estimator__alpha': [100, 10, 1, 0.1, 1e-2, 1e-3]}

elastic = OneVsRestClassifier(ElasticNet())
param_elastic = {'estimator__alpha': [100, 10, 1, 0.1, 1e-2, 1e-3], 
                 'estimator__l1_ratio': np.linspace(0.01, 0.99, 5)}

ridge = RidgeClassifier(solver='lsqr')
param_ridge = {'alpha': [100, 10, 1, 0.1, 1e-2, 1e-3]}

huber = OneVsRestClassifier(HuberSVC())
param_huber = {'estimator__C': [100, 10, 1, 0.1, 1e-2, 1e-3], 
              'estimator__lambd': [100, 10, 1, 0.1, 1e-2, 1e-3], 
              'estimator__mu': [100, 10, 1, 0.1, 1e-2, 1e-3]}

n_folds = 5
param_folds = 3
scoring = 'accuracy'

result_df = pandas.DataFrame()
for i, (train_index, test_index) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
    for clf_name, clf, param_grid in [('Lasso', lasso, param_lasso), 
                                      ('ElasticNet', elastic, param_elastic), 
                                      ('Ridge', ridge, param_ridge), 
                                      ('HuberSVC', huber, param_huber)]:

        gs = GridSearchCV(clf, param_grid, scoring=scoring, cv=param_folds, n_jobs=-1)
        gs.fit(X[train_index], y[train_index])
        best_clf = gs.best_estimator_

        score = accuracy_score(y[test_index], best_clf.predict(X[test_index]))
        result_df.loc[i, clf_name] = score

print "result shows: \n"
result_df.loc['Mean'] = result_df.mean()
pandas.options.display.float_format = '{:,.3f}'.format
result_df
print result_df.values
