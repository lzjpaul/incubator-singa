import numpy as np
import sys
sys.path.append("..")

from ktensor import ktensor

R = 10
A1 = 100
A2 = 80
A3 = 50

A = ktensor(np.ones(R), [np.random.rand(A1, R), np.random.rand(A2, R),
                         np.random.rand(A3, R)])
B = ktensor(np.ones(R), [np.random.rand(A1, R), np.random.rand(A2, R),
                         np.random.rand(A3, R)])
rawFMS = A.fms(B)
greedFMS = A.greedy_fms(B)
print rawFMS, greedFMS

np.random.seed(10)
A = ktensor(np.ones(R), [np.random.randn(5, R),
                         np.random.randn(5, R),
                         np.random.randn(2, R)])
A.U = [np.multiply((A.U[n] > 0).astype(int), A.U[n])
       for n in range(A.ndims())]
B = ktensor(np.ones(R), [np.random.randn(5, R),
                         np.random.randn(5, R),
                         np.random.randn(2, R)])
B.U = [np.multiply((B.U[n] > 0).astype(int), B.U[n]) for n in range(B.ndims())]

rawFOS = A.fos(B)
greedFOS = A.greedy_fos(B)
print rawFOS, greedFOS

A2 = 90
A3 = 80
A = ktensor(np.ones(R), [np.random.rand(A1, R), np.random.rand(A2, R),
                         np.random.rand(A3, R)])
B = ktensor(np.ones(R), [np.random.rand(A1, R), np.random.rand(A2, R),
                         np.random.rand(A3, R)])

R = 20
tmpA = ktensor(np.ones(R), [np.random.rand(A1, R), np.random.rand(A2, R),
                         np.random.rand(A3, R)])
print tmpA.norm()
print (A - B).norm()
