'''
Base code from pytensor: the python implementation of MATLAB based tensor code
https://code.google.com/p/pytensor

A sparse representation of the tensor object, storing only nonzero entries.
The code is the python implementation of the @sptensor folder in the
MATLAB Tensor Toolbox
'''
import numpy as np
import cmath
from sptenmat import sptenmat
from tenmat import tenmat
from tensor import tensor
from dtensor import dtensor
from scipy import sparse
import tools


def check_value(value):
    """ Check validity of value """
    err = ValueError("Subscripts must be a matrix of non-negative integers")
    if cmath.isnan(value):
        raise err
    if cmath.isinf(value):
        raise err
    if value < 0:
        raise err
    if value != round(value):
        raise err
    return


def tt_sizecheck(size):
    """Check whether the given size is valid. Used for sptensor"""
    size = np.array(size)
    if size.ndim != 1:
        raise ValueError("size must be a row vector of real positive integers")
    [check_value(val) for val in size]
    return True


def tt_subscheck(subs):
    """
    Check whether the given list of subscripts are valid.
    Used for sptensor """
    if subs.size == 0:
        return True
    if subs.ndim != 2:
        raise ValueError("Subscript dimensions is incorrect")
    for i in range(0, (subs.size / subs[0].size)):
        for j in range(0, (subs[0].size)):
            val = subs[i][j]
            check_value(val)
    return True


def tt_valscheck(vals):
    """Check whether the given list of values are valid. Used for sptensor"""
    if vals.size == 0:
        return True
    if vals.ndim != 2 or vals[0].size != 1:
        raise ValueError("values must be a column array")
    return True


class sptensor(tensor):
    subs = None
    vals = None
    func = None

    def __init__(self,
                 subs,
                 vals,
                 shape=None,
                 func=sum.__call__):
        """
        Create a sptensor object. The subs array specifies the nonzero entries
        in the tensor, with the kth row of subs corresponding to the kth entry
        in vals.

        Parameters
        ----------
        subs - p x n array specifying the subscripts of nonzero entries
        vals - the corresponding value of the nonzero entries
        shape - the shape of the tensor object.
        func - accumulation function for repeated subscripts

        Returns
        -------
        out : sparse tensor object

        """
        if isinstance(subs, list):
            subs = np.array(subs)
        if isinstance(vals, list):
            vals = np.array(vals)
        if isinstance(shape, list):
            shape = np.array(shape)
        if not tt_subscheck(subs):
            raise ValueError("Error in subscripts")
        if not tt_valscheck(vals):
            raise ValueError("Error in values")
        if shape is not None and not tt_sizecheck(shape):
            raise ValueError("Error in shape")
        if(vals.size != 0 and vals.size != 1 and len(vals) != len(subs)):
            raise ValueError("Number of subscripts and values must be equal")

        if shape is None:
            self.shape = tuple(subs.max(0) + 1)
        else:
            self.shape = tuple(shape)
        self.func = func

        if subs.size == 0:
            self.vals = np.array([])
            self.subs = np.array([])
        else:
            (newsub, loc) = uniquerows(subs)
            newval = np.zeros([len(newsub), 1])
            for i in range(0, len(loc)):
                newval[(int)(loc[i])] = func(vals[i], newval[(int)(loc[i])])

            nnzIdx = np.flatnonzero(newval)
            self.vals = newval[nnzIdx, :]
            self.subs = np.array(newsub.tolist())[nnzIdx, :]

    def mttkrp(self, U, n):
        """
        Matricized tensor times Khatri-Rao product for sparse tensor.

        Calculates the matrix product of the n-mode matricization of X with
        the Khatri-Rao product of all entries in U except the nth.
        A series of TTV operations are performed rather than forming the
        Khatri-Rao product.

        Parameters
        ----------
        U - factorization
        n - the mode not to calculate

        Returns
        -------
        out : Khatri-Rao product as a numpy array
        """
        N = self.ndims()
        if (n == 0):
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]
        V = np.zeros((self.shape[n], R))

        for r in range(R):
            Z = [{} for i in range(N)]
            dim = np.concatenate((np.arange(0, n), np.arange(n + 1, N)))
            for i in dim:
                Z[i] = U[i][:, r]
            V[:, r] = self.ttv(Z, dim).tondarray()
        return V

    def norm(self):
        """ returns the Frobenius norm of the tensor."""
        return np.linalg.norm(self.vals)

    def nnz(self):
        """returns the number of non-zero elements in the sptensor"""
        return len(self.subs)

    def to_dtensor(self):
        """returns a new tensor object that contains the same values"""
        temp = np.zeros(self.shape)
        idx = np.ravel_multi_index(self.subs.T, self.shape)
        temp.put(idx, self.vals.flatten())
        return dtensor(temp, self.shape)

    def __str__(self):
        if (self.nnz() == 0):
            return "Empty sparse tensor of size {0}".format(self.shape)
        else:
            ret = "sparse tensor of size {0} with \
                  {1} non-zero elements\n".format(self.shape, self.nnz())
            for i in range(len(self.subs)):
                ret += "\n{0} {1}".format(self.subs[i], self.vals[i])
            return ret

    def copy(self):
        return sptensor(self.subs.copy(), self.vals.copy(),
                        self.shape, self.func)

    def permute(self, order):
        """returns a new sptensor permuted by the given order"""
        if (order.__class__ == list):
            order = np.array(order)

        if(self.ndims() != len(order)):
            raise ValueError("invalid permutation order")

        sortedorder = order.copy()
        sortedorder.sort()
        if not ((sortedorder == np.arange(len(self.shape))).all()):
            raise ValueError("invalid permutation order")
        neworder = np.arange(len(order)).tolist()
        newsiz = list(self.shape)
        newval = self.vals.copy()
        newsub = self.subs.copy()

        for i in range(len(order) - 1):
            index = tools.find(neworder, order[i])
            for s in newsub:
                temp = s[i]
                s[i] = s[index]
                s[index] = temp
            temp = newsiz[i]
            newsiz[i] = newsiz[index]
            newsiz[index] = temp

            temp = neworder[i]
            neworder[i] = neworder[index]
            neworder[index] = temp

        return sptensor(newsub, newval, newsiz, self.func)

    def ttm(self, mat, dims=None, option=None):
        """
        Computes the sptensor times the given matrix.
        arrs is a single 2-D matrix/array or a list of those matrices/arrays.
        """
        if dims is None:
            dims = range(self.ndims())
        # Handle when arrs is a list of arrays
        if isinstance(mat, list):
            if(len(mat) == 0):
                raise ValueError("the given list of arrays is empty!")
            (dims, vidx) = tools.tt_dimscheck(dims, self.ndims(), len(mat))

            Y = self.ttm(mat[vidx[0]], dims[0], option)
            for i in range(1, len(dims)):
                Y = Y.ttm(mat[vidx[i]], dims[i], option)
            return Y
        if(mat.ndim != 2):
            raise ValueError("matrix in 2nd argument must be a matrix!")

        if option is not None:
            if (option == 't'):
                mat = mat.transpose()
            else:
                raise ValueError("unknown option.")
        if(dims.__class__ == list):
            if(len(dims) != 1):
                raise ValueError("Error in number of elements in dims")
            else:
                dims = dims[0]
        if(dims < 0 or dims > self.ndims()):
            raise ValueError("Dimension N not between 1 and num of dimensions")
        # Check that sizes match
        if(self.shape[dims] != mat.shape[1]):
            raise ValueError("size mismatch on V")

        # Compute the new size
        newsiz = list(self.shape)
        newsiz[dims] = mat.shape[0]

        # Compute Xn
        Xnt = sptenmat.sptenmat(self, None, [dims], None, 't')
        rdims = Xnt.rdims
        cdims = Xnt.cdims

        I = []
        J = []
        for i in range(0, len(Xnt.subs)):
            I.extend([Xnt.subs[i][0]])
            J.extend([Xnt.subs[i][1]])

        Z = sparse.coo_matrix((Xnt.vals.flatten(), (I, J)),
                              shape=(tools.getelts(Xnt.tsize,
                                                   Xnt.rdims).prod(),
                                     tools.getelts(Xnt.tsize,
                                                   Xnt.cdims).prod()))
        Z = Z * mat.T

        Z = dtensor.dtensor(Z, newsiz).tosptensor()

        if Z.nnz() <= 0.5 * np.array(newsiz).prod():
            Ynt = sptenmat.sptenmat(Z, rdims, cdims)
            return Ynt.tosptensor()
        else:
            Ynt = tenmat.tenmat(Z.to_dtensor(), rdims, cdims)
            return Ynt.totensor()

    def ttv(self, v, dims):
        """
        Computes the product of this tensor with the column vector along
        specified dimensions.

        Parameters
        ----------
        v - column vector
        d - dimensions to multiply the product

        Returns
        -------
        out : a sparse tensor if 50% or fewer nonzeros
        """

        (dims, vidx) = tools.tt_dimscheck(dims, self.ndims(), len(v))
        remdims = np.setdiff1d(range(self.ndims()), dims)
        newvals = self.vals
        subs = self.subs

        # Multiple each value by the appropriate elements
        # of the corresponding vector
        for n in range(len(dims)):
            idx = subs[:, dims[n]]  # extract indices for dimension n
            w = v[vidx[n]]         # extract nth vector
            bigw = w[idx]          # stretch out the vector
            newvals = np.multiply(newvals.flatten(), bigw)

        # Case 0: all dimensions specified - return the sum
        if len(remdims) == 0:
            return np.sum(newvals)
        # Otherwise figure out the subscripts and accumulate the results
        newsubs = self.subs[:, remdims]
        newsiz = np.array(self.shape)[remdims]
        # Case 1: return a vector
        if len(remdims) == 1:
            c = tools.accum_np(newsubs, newvals, newsiz[0])
            if np.count_nonzero(c) < 0.5 * newsiz[0]:
                c = sptensor(np.arange(newsiz[0]).reshape(newsiz[0], 1),
                             c.reshape(newsiz[0], 1))
            else:
                c = dtensor(c, newsiz)
            return c
        # Case 2: result is a multi-way array
        c = sptensor(newsubs, newvals.reshape(len(newvals), 1), newsiz)
        # check to see if it's dense
        if c.nnz() > 0.5 * np.prod(c.shape):
            return c.to_dtensor()
        return c

    def tondarray(self):
        """returns an ndarray that contains the data of the sptensor"""
        return self.to_dtensor().tondarray()

    def save(self, outfile):
        np.save(outfile, self.subs)
        np.save(outfile, self.vals)
        np.save(outfile, self.shape)

    def _checkshape(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of the tensors do not match")

    def _check_object(self, other):
        if isinstance(other, sptensor):
            return True
        elif isinstance(other, dtensor):
            return False
        raise NotImplementedError("Can only handle same dtensor object")

    def __add__(self, other):
        self._checkshape(other)
        if self._check_object(other):
            return sptensor(self.subs.tolist() + other.subs.tolist(),
                            self.vals.tolist() + other.vals.tolist(),
                            self.shape)
        # other is a tensor or a scalar value
        return self.to_dtensor() + other

    def __sub__(self, other):
        self._checkshape(other)
        if self._check_object(other):
            return sptensor(self.subs.tolist() + other.subs.tolist(),
                            self.vals.tolist() + (-other.vals).tolist(),
                            self.shape)
        # other is a tensor or a scalar value
        return self.to_dtensor() - other

    def __eq__(self, other):
        self._checkshape(other)
        if self._check_object(other):
            sub1 = self.subs
            sub2 = other.subs
            usub = union(sub1, sub2)
            # ret = (tools.allIndices(other.shape))
        elif isinstance(other, dtensor):
            return other.__eq__(self.to_dtensor())
        # elif(other.__class__ == int or other.__class__ == float or oth.__class__ == bool):
        #     newvals = (self.vals == other)
        #     newvals = booltoint(newvals)
        #     return sptensor(self.subs, newvals, self.size)
        else:
            raise ValueError("Incomparable Types")

    def __mul__(self, other):
        if isinstance(other, (int, long, float)):
            return sptensor(self.subs.copy(), self.vals.copy() * other,
                            self.shape)
        raise ValueError("Use ttt() instead.")

    def __neg__(self):
        return sptensor(self.subs.copy(), self.vals.copy() * -1, self.shape)


def load(infile):
    """
    Load a tensor from a file that has been saved using sptensor.save()
    """
    subs = np.load(infile)
    vals = np.load(infile)
    siz = np.load(infile)
    return sptensor(subs, vals, siz)


def diag(vals, shape=None):
    """
    Special constructor
    A sptensor with the given values in the diagonal
    """
    # if shape is None or
    # number of dimensions of shape is less than the number of values given
    if shape is None or len(shape) < len(vals):
        shape = [len(vals)] * len(vals)
    else:
        shape = list(shape)
        for i in range(0, len(vals)):
            if(shape[i] < len(vals)):
                shape[i] = len(vals)
    subs = []
    for i in range(0, len(vals)):
        subs.extend([[i] * len(shape)])

    vals = np.array(vals).reshape([len(vals), 1])
    return sptensor(subs, vals, shape)


def uniquerows(arr):
    """
    Given a 2D array, find the unique row and return the rows as 2-d array.
    """
    arr_dtype = arr.dtype.descr * arr.shape[1]
    struct = arr.view(arr_dtype)
    arr_uniq, idx = np.unique(struct, return_inverse=True)
    return (arr_uniq, idx)


# arr1, arr2: sorted list or sorted numpy.ndarray of subscripts.
# union returns the sorted union of arr1 and arr2.
def union(arr1, arr2):
    if(arr1.__class__ != list):
        a1 = arr1.tolist()
    else:
        a1 = arr1
    if(arr2.__class__ != list):
        a2 = arr2.tolist()
    else:
        a2 = arr1

    i = 0
    j = 0
    ret = np.array([])

    if len(a1):
        ret = [a1[i]]
        i = i + 1
    elif(len(a2) > 0):
        ret = [a2[j]]
        j = j + 1
    else:
        return np.array([[]])

    while(i < len(a1) or j < len(a2)):
        if(i == len(a1)):
            ret = np.concatenate((ret, [a2[j]]), axis=0)
            j = j + 1
        elif(j == len(a2)):
            ret = np.concatenate((ret, [a1[i]]), axis=0)
            i = i + 1
        elif(a1[i] < a2[j]):
            ret = np.concatenate((ret, [a1[i]]), axis=0)
            i = i + 1
        elif(a1[i] > a2[j]):
            ret = np.concatenate((ret, [a2[j]]), axis=0)
            j = j + 1
        else:
            i = i + 1

    return ret


def tosptensor(tensor):
    """ returns the sptensor object
    that contains the same value with the tensor object."""
    nnz = np.nonzero(tensor._data)
    vals = tensor._data[nnz]
    totVals = len(vals)
    vals = np.reshape(vals, (totVals, 1))
    subs = np.empty(shape=(totVals, 0), dtype='int')
    for idx in nnz:
        subs = np.column_stack((subs, idx))
    return sptensor(subs, vals, tensor.shape)
