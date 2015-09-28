#!/usr/bin/env python

import numpy as np
from tenmat import tenmat
from dtensor import dtensor


def ctor(verbose):
    dat = np.arange(24).reshape([2, 3, 4])
    t = dtensor(dat)
    print t
    if (verbose):
        obj = tenmat(t, [1, 0])
        print obj
        print obj.copy()
    dat = dat.reshape([4, 6])
    t = dtensor(dat)
    if verbose:
        obj = tenmat(t, [0], [1], [4, 6])
        print obj


def totensorTests(verbose):
    dat = np.arange(24).reshape([2, 3, 4])
    t = dtensor(dat)
    obj = tenmat(t, [2, 1])
    if verbose:
        print obj
        print obj.totensor()
