import numpy as np
import sys
sys.path.append("..")

import dtensor
import sptensor
import CP_ALS

"""
Test file associated with the CP decomposition using ALS
"""
X = dtensor.dtensor(range(1, 25), [3, 4, 2])
print CP_ALS.cp_als(X, 2)

subs = np.array([[0, 3, 1], [1, 0, 1], [1, 2, 1],
                 [1, 3, 1], [3, 0, 0]])
vals = np.array([[1], [1], [1], [1], [3]])
siz = np.array([5, 5, 2])
X = sptensor.sptensor(subs, vals, siz)
print CP_ALS.cp_als(X, 2)
