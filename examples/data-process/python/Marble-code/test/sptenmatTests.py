#!/usr/bin/env python
import numpy
import sys
sys.path.append("..")

import sptensor
import sptenmat
import tenmat
import dtensor


def ctor(verbose):
    x = numpy.array([[[0, 0, 0.9052], [0.9121, 0, 0.7363]],
                     [[0.1757, 0.2089, 0], [0, 0.7455, 0]],
                     [[0, 0, 0.6754], [0, 0, 0]]])
    obj = sptenmat.sptenmat(x, [0], [1, 2], [10, 10, 10])
    print obj
    subs = numpy.array([[1, 3, 5], [1, 1, 0], [2, 2, 2],
                        [3, 4, 4], [1, 1, 1], [1, 1, 1]])
    vals = numpy.array([[0.5], [1.5], [100], [3.5], [4.5], [5.5]])
    siz = numpy.array([4, 5, 6])
    spt = sptensor.sptensor(subs, vals, siz)
    print spt
    obj = sptenmat.sptenmat(spt, [0, 1], [2])
    print obj


def tosptensorTest():
    subs = numpy.array([[1, 3, 5], [1, 1, 0], [2, 2, 2], [3, 4, 4],
                        [1, 1, 1], [1, 1, 1]])
    vals = numpy.array([[0.5], [1.5], [100], [3.5], [4.5], [5.5]])
    siz = numpy.array([4, 5, 6])
    spt = sptensor.sptensor(subs, vals, siz)
    print spt
    sptm = sptenmat.sptenmat(spt, [1])
    print sptm
    tm = tenmat.tenmat(spt.to_dtensor(), [1])
    print tm
    temp = sptm.tosptensor()
    print temp


def compareTensor():
    dat = numpy.arange(24).reshape([3, 4, 2], order='F')
    t = dtensor.dtensor(dat)
    vals = numpy.reshape(numpy.arange(24), (24, 1))
    subs = numpy.zeros((24, 3), dtype=int)
    subs[:, 0] = numpy.tile(numpy.arange(3), 8)
    subs[:, 1] = numpy.tile(numpy.repeat(numpy.arange(4), 3), 2)
    subs[:, 2] = numpy.repeat(numpy.arange(2), 12)
    spt = sptensor.sptensor(subs, vals, (3, 4, 2))
    ## doublecheck they are equal
    print (t == spt.to_dtensor()).all()
    print tenmat.tenmat(t, [0])
    print (sptenmat.sptenmat(spt, [0])).data.toarray()
    print tenmat.tenmat(t, [1])
    print (sptenmat.sptenmat(spt, [1])).data.toarray()
    print tenmat.tenmat(t, [2])
    print (sptenmat.sptenmat(spt, [2])).data.toarray()
