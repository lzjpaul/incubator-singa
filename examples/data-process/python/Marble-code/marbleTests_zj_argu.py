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
print "open subs1\n"
file = open(sys.argv[2])
subs = np.genfromtxt(file, delimiter=",")
subs = subs.astype(np.int)
print "open vlas1\n"
file = open(sys.argv[3])
vals = np.genfromtxt(file, delimiter=",")
vals = np.matrix(vals).T
vals = vals.astype(np.int)
file = open(sys.argv[4])
siz = np.genfromtxt(file, delimiter=",")
siz = siz.astype(np.int)
print "subs.shape = \n", subs.shape
print "vals.shape = \n", vals.shape
print "siz.shape = \n", siz.shape
print "siz = \n", siz

X = sptensor.sptensor(subs, vals, siz)

rank = int(sys.argv[1])
print "rank num = \n", rank
maxiternum = int(sys.argv[5])
print "maxiternum = \n", maxiternum
marble = MarbleAPR(X, rank, 10000);

iterInfo, ll = marble.compute_decomp(gamma = [0.0001, 0.01, 0.01], gradual = True, max_inner = 15, max_iter = maxiternum, del_tol = 1e-20)

# signal_factors = marble.get_signal_factors()
# print "signal_factors = \n\n", signal_factors

file = open(sys.argv[6])
subs2 = np.genfromtxt(file, delimiter=",")
subs2 = subs2.astype(np.int)
file = open(sys.argv[7])
vals2 = np.genfromtxt(file, delimiter=",")
vals2 = np.matrix(vals2).T
vals2 = vals2.astype(np.int)
file = open(sys.argv[8])
siz2 = np.genfromtxt(file, delimiter=",")
siz2 = siz2.astype(np.int)
print "subs2.shape = \n", subs2.shape
print "vals2.shape = \n", vals2.shape
print "siz2.shape = \n", siz2.shape
print "siz2 = \n", siz2

Xhat = sptensor.sptensor(subs2, vals2, siz2)

np.random.seed(10)
projMat_test, biasMat_test = marble.project_data(Xhat, 0, max_iter = maxiternum, max_inner = 15, delta_tol = 1e-20)
projMat_train, biasMat_train = marble.project_data(X, 0, max_iter = maxiternum, max_inner = 15, delta_tol = 1e-20)
# print "projMat = \n\n", projMat
# print "biasMat = \n\n", biasMat

file = open(sys.argv[10], "w")
np.savetxt(file, projMat_test, "%f", ",")
file.close()
file = open(sys.argv[9], "w")
np.savetxt(file, projMat_train, "%f", ",")
file.close()
