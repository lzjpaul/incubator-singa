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
file = open("/data/zhaojing/marble/tensor/subs1.csv")
subs = np.genfromtxt(file, delimiter=",")
subs = subs.astype(np.int)
print "open vlas1\n"
file = open("/data/zhaojing/marble/tensor/vals1.csv")
vals = np.genfromtxt(file, delimiter=",")
vals = np.matrix(vals).T
vals = vals.astype(np.int)
file = open("/data/zhaojing/marble/tensor/siz.csv")
siz = np.genfromtxt(file, delimiter=",")
siz = siz.astype(np.int)
print "subs.shape = \n", subs.shape
print "vals.shape = \n", vals.shape
print "siz.shape = \n", siz.shape
print "siz = \n", siz

X = sptensor.sptensor(subs, vals, siz)

marble = MarbleAPR(X, 75, 0.1);

iterInfo, ll = marble.compute_decomp(gamma = [0.0001, 0.01, 0.01], gradual = True, max_inner = 30, max_iter = 90, del_tol = 1e-10)

# signal_factors = marble.get_signal_factors()
# print "signal_factors = \n\n", signal_factors

file = open("/data/zhaojing/marble/tensor/subs2.csv")
subs2 = np.genfromtxt(file, delimiter=",")
subs2 = subs2.astype(np.int)
file = open("/data/zhaojing/marble/tensor/vals2.csv")
vals2 = np.genfromtxt(file, delimiter=",")
vals2 = np.matrix(vals2).T
vals2 = vals2.astype(np.int)
file = open("/data/zhaojing/marble/tensor/siz.csv")
siz2 = np.genfromtxt(file, delimiter=",")
siz2 = siz2.astype(np.int)
print "subs2.shape = \n", subs2.shape
print "vals2.shape = \n", vals2.shape
print "siz2.shape = \n", siz2.shape
print "siz2 = \n", siz2

Xhat = sptensor.sptensor(subs2, vals2, siz2)

np.random.seed(10)
projMat_test, biasMat_test = marble.project_data(Xhat, 0, max_iter = 90, max_inner = 30, delta_tol = 1e-10)
projMat_train, biasMat_train = marble.project_data(X, 0, max_iter = 90, max_inner = 30, delta_tol = 1e-10)
# print "projMat = \n\n", projMat
# print "biasMat = \n\n", biasMat

file = open("/data/zhaojing/marble/tensor/membership_test_75_50_90.txt", "w")
np.savetxt(file, projMat_test, "%f", ",")
file.close()
file = open("/data/zhaojing/marble/tensor/membership_train_75_50_90.txt", "w")
np.savetxt(file, projMat_train, "%f", ",")
file.close()
