import numpy as np
import sys

import CP_APR
import ktensor
import dtensor
import sptensor
from marbleAPR import MarbleAPR
"""
Test file associated with the MARBLE decomposition using APR
"""

""" Test factorization of sparse tensor """
subs = np.array([[0, 3, 1], [1, 0, 1], [1, 2, 1], [1, 3, 1], [3, 0, 0]])
vals = np.array([[1], [1], [1], [1], [3]])
siz = np.array([10, 5, 2])  # 4x5x2 tensor
X = sptensor.sptensor(subs, vals, siz)

marble = MarbleAPR(X, 5, 0.1);

iterInfo, ll = marble.compute_decomp(max_inner = 10, max_iter = 10, del_tol = 0.01)

# signal_factors = marble.get_signal_factors()
# print "signal_factors = \n\n", signal_factors
subs2 = np.array([[0, 3, 1], [3, 2, 0], [3, 2, 1]])
vals2 = np.array([[1], [3], [3]])
siz2 = np.array([10, 5, 2])
Xhat = sptensor.sptensor(subs2, vals2, siz2)

np.random.seed(10)
projMat, biasMat = marble.project_data(Xhat, 0, max_iter = 10, max_inner = 10, delta_tol = 0.01)
print "projMat = \n\n", projMat
print "biasMat = \n\n", biasMat
