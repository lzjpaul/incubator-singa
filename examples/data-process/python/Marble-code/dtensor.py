'''
Base code from pytensor: the python implementation of MATLAB based tensor code
https://code.google.com/p/pytensor

The regular, dense tensor object.
The code is the python implementation of the MATLAB Tensor Toolbox
'''
import numpy as np
from tensor import tensor
import tools
import khatrirao


class dtensor(tensor):
    _data = None

    def __init__(self, data, shape=None):
        """Constructor for tensor object.
        data can be numpy.array or list.
        shape can be numpy.array, list, tuple of integers"""
        if isinstance(data, list):
            data = np.array(data)
        # perform sanity check of the shape
        if shape is None:
            shape = tuple(data.shape)
        elif len(shape) == 0:
            raise ValueError("Second argument must be a row vector.")
        elif isinstance(shape, np.ndarray):
            if shape.ndim != 2 and shape[0].size != 1:
                raise ValueError("Second argument must be a row vector.")
        else:
            shape = tuple(shape)

        if len(shape) == 0 and data.size != 0:
            raise ValueError("Empty tensor cannot contain any elements")
        if np.prod(np.array(shape)) != data.size:
            raise ValueError("Data size does not match size of tensor")
        self.shape = shape
        self._data = data.reshape(self.shape, order='F')

    def copy(self):
        return dtensor(self._data.copy(), self.shape)

    def mttkrp(self, U, n):
        N = self.ndims()
        if len(U) != N:
            raise ValueError("U has the wrong length")
        Xn = self.permute(np.concatenate(([n], np.arange(0, n),
                                          np.arange(n + 1, N))))
        # use the Fortran ordering system for consistency w/ Matlab
        Xn = Xn._data.reshape(self.dimsize(n),
                              np.prod(self.shape) / self.dimsize(n),
                              order='F')
        Z = khatrirao.khatrirao_array([U[i] for i in range(len(U)) if i != n],
                                      reverse=True)
        V = np.dot(Xn, Z)
        return V

    def norm(self):
        """ returns the Frobenius norm of the tensor."""
        return np.linalg.norm(self._data.flatten())

    def permute(self, order):
        """ returns a tensor permuted by the order specified. """
        if isinstance(order, list):
            order = np.array(order)
        if self.ndims() != len(order):
            raise ValueError("Invalid permutation order")

        sortedorder = order.copy()
        sortedorder.sort()
        if not ((sortedorder == np.arange(self._data.ndim)).all()):
            raise ValueError("Invalid permutation order")
        neworder = np.arange(len(order)).tolist()
        newshape = list(self.shape)
        newdata = self._data.copy()
        for i in range(len(order) - 1):
            index = tools.find(neworder, order[i])
            newdata = newdata.swapaxes(i, index)
            temp = newshape[i]
            newshape[i] = newshape[index]
            newshape[index] = temp
            temp = neworder[i]
            neworder[i] = neworder[index]
            neworder[index] = temp
        newshape = tuple(newshape)
        return dtensor(newdata, newshape)

    def ipermute(self, order):
        """
        returns a tensor permuted by the inverse of the order specified.
        """
        # calculate the inverse of iorder
        iorder = [[tools.find(order, i)] for i in range(len(order))]
        # returns the permuted tensor by the inverse
        return self.permute(iorder)

    def ttm(self, mat, dims=None, option=None):
        """
        Computes the tensor times the given matrix.
        arrs is a single 2-D matrix/array or a list of those matrices/arrays.
        """
        if dims is None:
            dims = range(self.ndims())
        # handle a list of arrays
        if isinstance(mat, list):
            if len(mat) == 0:
                raise ValueError("the given list of arrays is empty!")
            (dims, vidx) = tools.tt_dimscheck(dims, self.ndims(), len(mat))
            Y = self.ttm(mat[vidx[0]], dims[0], option)
            for i in range(1, len(dims)):
                Y = Y.ttm(mat[vidx[i]], dims[i], option)
            return Y
        if mat.ndim != 2:
            raise ValueError("matrix in 2nd argument must be a matrix!")
        if isinstance(dims, list):
            if len(dims) != 1:
                raise ValueError("Error in number of elements in dims")
            else:
                dims = dims[0]
        if dims < 0 or dims > self.ndims():
            raise ValueError("Dimension N must be between 1 and number of dimensions")
        
        #Compute the product
        N = self.ndims()
        shp = self.shape
        order = []
        order.extend([dims])
        order.extend(range(dims))
        order.extend(range(dims + 1, N))
        
        newdata = self.permute(order)._data
        newdata = newdata.reshape(shp[dims], np.prod(np.array(shp)) / shp[dims])
        if option is None:
            newdata = np.dot(mat, newdata)
            p = mat.shape[0]
        elif(option == 't'):
            newdata = np.dot(mat.transpose(), newdata)
            p = mat.shape[1]
        else:
            raise ValueError("Unknown option")
        
        newshp = [p]
        newshp.extend(tools.getelts(shp, range(dims)))
        newshp.extend(tools.getelts(shp, range(dims + 1, N)))
        
        Y = dtensor(newdata, newshp)
        Y = Y.ipermute(order)
        return Y

    def ttv(self, v, dims):
        (dims, vidx) = tools.tt_dimscheck(dims, self.ndims(), len(v))
        remdims = np.setdiff1d(range(self.ndims()), dims)
        if self.ndims() > 1:
            c = self.permute(np.concatenate((remdims, dims)))._data
        
        n = self.ndims() - 1
        sz = np.array(self.shape)[np.concatenate((remdims, dims))]
        for i in range(len(dims) - 1, -1, -1):
            c = c.reshape(np.prod(sz[0:n]), sz[n], order='F')
            c = np.dot(c, v[vidx[i]])
            n = n - 1
        if n > 0:
            c = dtensor(c, sz[0:n])
        else:
            c = c[0]
        return c
        
    def tondarray(self):
        """return an ndarray that contains the data of the tensor"""
        return self._data
    
    def _check_objects(self, other):
        if not isinstance(other, dtensor):
            raise NotImplementedError("Can only handle same dtensor object")
        if self.shape != other.shape:
            raise ValueError("Shapes of the tensors do not match")

    def save(self, filename):
        raise NotImplementedError("Save not implemented")

    # Math, logic operators
    def __neg__(self):
        return dtensor(self._data * -1, self.shape)

    def __add__(self, other):
        self._check_objects(other)
        return dtensor(self._data.__add__(other._data), self.shape)

    def __sub__(self, other):
        self._check_objects(other)
        return dtensor(self._data.__sub__(other._data), self.shape)

    def __mul__(self, other):
        self._check_objects(other)
        raise ValueError("Use ttt() instead.")

    def __eq__(self, other):
        self._check_objects(other)
        return np.equal(self._data, other._data)

    def __str__(self):
        desc = "tensor of size {0}\n".format(self.shape)
        desc += self._data.__str__()
        return desc
        

#Special Constructors
def zeros(shape_):
    """
    Special constructor
    A tensor filled with 0
    """
    data = np.zeros(shape_)
    return dtensor(data, shape_)


def ones(shape_):
    """
    Special constructor
    A tensor filled with 1
    """
    data = np.ones(shape_)
    return dtensor(data, shape_)


def random(shape_):
    """
    Special constructor
    A tensor filled with random number between 0 and 1
    """
    data = np.random.random(shape_)
    return tensor(data, shape_)


def diag(vals, shape=None):
    """
    Special constructor
    A tensor with the values in the diagonal
    """
    #if shape is None or
    #number of dimensions of shape is less than the number of values given
    if shape is None or len(shape) < len(vals):
        shape = [len(vals)] * len(vals)
    else:
        shape = list(shape)
        for i in range(0, len(vals)):
            if(shape[i] < len(vals)):
                shape[i] = len(vals)
    data = np.zeros(shape)
    # put the values in the ndarray
    for i in range(len(vals)):
        data.put(tools.sub2ind(shape, [i] * len(shape)), vals[i])
    return tensor(data, shape)
    