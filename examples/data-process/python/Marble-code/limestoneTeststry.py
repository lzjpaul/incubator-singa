import numpy as np
import sys

import sptensor
import ktensor
from limestone import Limestone

"""
Test file for limestone
"""

""" Test factorization of sparse matrix """
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

lstone = Limestone(X, 4)
lstone.decompose(Minit=Minit, maxiters=100)

subs2 = np.array([[0, 3, 1], [1, 2, 0]])
vals2 = np.array([[1], [1]])
siz2 = np.array([2, 5, 2])
Xhat = sptensor.sptensor(subs2, vals2, siz2)

np.random.seed(10)
norm_rows = lstone.projectData(X, 2)
print "limestone norm_rows = \n", norm_rows
