import numpy as np
import sys
sys.path.append("..")

import CP_APR
import ktensor
import dtensor
import sptensor

"""
Test file associated with the CP decomposition using APR
"""

""" Test factorization of sparse tensor """
subs = np.array([[0, 3, 1], [1, 0, 1], [1, 2, 1], [1, 3, 1], [3, 0, 0]])
vals = np.array([[1], [1], [1], [1], [3]])
siz = np.array([5, 5, 2])  # 5x5x2 tensor
X = sptensor.sptensor(subs, vals, siz)
U0 = np.array([[0.7689, 0.8843, 0.7487, 0.0900],
               [0.1673, 0.5880, 0.8256, 0.1117],
               [0.8620, 0.1548, 0.7900, 0.1363],
               [0.9899, 0.1999, 0.3185, 0.6787],
               [0.5144, 0.4070, 0.5341, 0.4952]])
U1 = np.array([[0.1897, 0.5606, 0.8790, 0.9900],
               [0.4950, 0.9296, 0.9889, 0.5277],
               [0.1476, 0.6967, 0.0006, 0.4795],
               [0.0550, 0.5828, 0.8654, 0.8013],
               [0.8507, 0.8154, 0.6126, 0.2278]])
U2 = np.array([[0.4981, 0.5747, 0.7386, 0.2467],
               [0.9009, 0.8452, 0.5860, 0.6664]])
Minit = ktensor.ktensor(np.ones(4), [U0, U1, U2])
fms = Minit.fms(Minit)

Y, _ = CP_APR.cp_apr(X, 4, Minit=Minit, maxiters=100)
Y.normalize_sort(1)

# """ Test factorization of 2-mode sparse matrix """
# matX = sptenmat.sptenmat(X, [0]).to_sptensor()
# CP_APR.cp_apr(matX, 4)

""" Test factorization of regular tensor """
XT = dtensor.dtensor(range(1, 25), [3, 4, 2])
CP_APR.cp_apr(XT, 4)
