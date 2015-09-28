'''
Base code from pytensor: the python implementation of MATLAB based tensor code
https://code.google.com/p/pytensor

Implementation of various common tensor functions.
The code is the python implementation of the various functions in the
MATLAB Tensor Toolbox
'''
import numpy as np
import math


def accum_np(accmap, a, size, func=np.sum):
    """
    Construct a numpy array with accumulation.
    The numpy equivalent to the matlab function accumarray.
    The elements are grouped and the function is applied to each group.
    """
    vals = [[] for i in range(size)]
    faccmap = accmap.flatten()
    for i in range(len(faccmap)):
        indx = faccmap[i]
        vals[indx].append(a[i])
    out = np.zeros(size)
    for i in range(size):
        out[i] = func(vals[i])
    return out


def allIndices(dim):
    """ From the given shape of dimenions (e.g. (2,3,4)),
    generate a numpy.array of all, sorted indices."""
    length = len(dim)
    sub = np.arange(dim[length - 1]).reshape(dim[length - 1], 1)
    for d in range(length - 2, -1, -1):
        for i in range(dim[d]):
            temp = np.ndarray([len(sub), 1])
            temp.fill(i)
            temp = np.concatenate((temp, sub), axis=1)
            if(i == 0):
                newsub = temp
            else:
                newsub = np.concatenate((newsub, temp), axis=0)
        sub = newsub
    return sub


def find(nda, obj):
    """
    Returns the index of the obj in the given nda(ndarray, list, or tuple)
    """
    for i in range(0, len(nda)):
        if(nda[i] == obj):
            return i
    return -1


def getelts(nda, indices):
    """
    From the given nda(ndarray, list, or tuple),
    returns the list located at the given indices
    """
    ret = []
    for i in indices:
        ret.extend([nda[i]])
    return np.array(ret)


def sub2ind(shape, subs):
    """ From the given shape, returns the index of the given subscript"""
    subs = np.array(subs)
    return np.ravel_multi_index(subs.T, shape).reshape((subs.shape[0], 1))


def ind2sub(shape, ind):
    """ From the given shape, returns the subscripts of the given index"""
    # subs = np.unravel_index(ind, shape)
    # return np.dstack(subs)
    revshp = []
    revshp.extend(shape)
    revshp.reverse()
    mult = [1]
    for i in range(0, len(revshp) - 1):
        mult.extend([mult[i] * revshp[i]])
    mult.reverse()
    mult = np.array(mult).reshape(len(mult))
    sub = []
    for i in range(0, len(shape)):
        sub.extend([math.floor(ind / mult[i])])
        ind = ind - (math.floor(ind / mult[i]) * mult[i])
    return sub


def tt_dimscheck(dims, N, M=None, exceptdims=False):
    """
    Checks whether the specified dimensions are valid in a
    tensor of N-dimension.
    If M is given, then it will also retuns an index for M multiplicands.
    If exceptdims == True, then it will compute for the
    dimensions not specified."""
    # if exceptdims is true
    if exceptdims:
        dims = listdiff(range(N), dims)
    #check vals in between 0 and N-1
    if any(x < 0 or x >= N for x in dims):
        raise ValueError("invalid dimensions specified")
    if M is not None and M > N:
        raise ValueError("Cannot have more multiplicands than dimensions")
    if M is not None and M != N and M != len(dims):
        raise ValueError("invalid number of multiplicands")
    # number of dimensions in dims
    p = len(dims)
    sdims = []
    sdims.extend(dims)
    sdims.sort()

    #indices of the elements in the sorted array
    sidx = []
    #table that denotes whether the index is used
    table = np.ndarray([len(sdims)])
    table.fill(0)

    for i in range(0, len(sdims)):
        for j in range(0, len(dims)):
            if(sdims[i] == dims[j] and table[j] == 0):
                sidx.extend([j])
                table[j] = 1
                break
    if M is None:
        return sdims
    if(M == p):
        vidx = sidx
    else:
        vidx = sdims
    return (sdims, vidx)


def listdiff(list1, list2):
    """returns the list of elements that are in list 1 but not in list2"""
    if isinstance(list1, np.ndarray):
        list1 = list1.tolist()
    if isinstance(list2, np.ndarray):
        list2 = list2.tolist()
    s = set(list2)
    return [x for x in list1 if x not in s]
