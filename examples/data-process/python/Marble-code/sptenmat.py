'''
Base code from pytensor: the python implementation of MATLAB based tensor code
https://code.google.com/p/pytensor

Matricization of a sparse tensor object.
The code is the python implementation of the @sptenmat folder
in the MATLAB Tensor Toolbox
'''

import numpy as np
from scipy import sparse


def _build_idx(subs, vals, dims, tshape):
    shape = np.array([tshape[d] for d in dims], ndmin=1)
    dims = np.array(dims, ndmin=1)
    if len(shape) == 0:
        idx = np.ones(len(vals), dtype=vals.dtype)
    elif len(subs) == 0:
        idx = np.array(tuple())
    else:
        idx = np.ravel_multi_index(tuple(subs[:, i] for i in dims), shape)
    return idx


class sptenmat:
    rdims = None
    cdims = None
    tsize = None
    data = None
    subs = None
    vals = None

    def __init__(self, T, rdim=None, cdim=None, tsiz=None, option=None):
        """Create a sptenmat object from a given ndarray or sptensor T"""
        if rdim is not None and isinstance(rdim, list):
            rdim = np.array(rdim)
        if cdim is not None and isinstance(cdim, list):
            cdim = np.array(cdim)
        if tsiz is not None and isinstance(tsiz, list):
            tsiz = np.array(tsiz)
        if rdim is not None:
            self.rdims = rdim
            if cdim is not None:
                self.cdims = cdim
            else:
                self.cdims = np.setdiff1d(range(len(T.shape)), rdim)
            M = np.prod([T.shape[r] for r in self.rdims])
            N = np.prod([T.shape[c] for c in self.cdims])
            ridx = _build_idx(T.subs, T.vals, self.rdims, T.shape)
            cidx = _build_idx(T.subs, T.vals, self.cdims[::-1], T.shape)

        self.data = sparse.coo_matrix((T.vals.flatten(),
                                       (ridx, cidx)),
                                      shape=(M, N))
        self.vals = T.vals
        self.subs = tuple((ridx, cidx))
        self.tsize = T.shape

    def to_coomat(self):
        """
        Returns a sparse coo matrix object(scipy.sparse) with the values
        """
        return self.data

    def to_csrmat(self):
        """
        Returns a compressed sparse row matrix
        """
        return self.data.tocsr()

    def __str__(self):
        ret = "sptenmat from an sptensor of size "
        ret += "{0} with {1} nonzeros\n".format(self.tsize, len(self.vals))
        ret += "rindices {0}\n".format(self.rdims)
        ret += "cindices {0}\n".format(self.cdims)
        ret += str(self.data)
        return ret
