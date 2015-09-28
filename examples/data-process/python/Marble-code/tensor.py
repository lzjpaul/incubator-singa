from abc import ABCMeta, abstractmethod
import numpy as np


class tensor(object):
    """
    Base tensor class from which all tensor classes are subclasses.
    Can not be instaniated
    See also
    --------
    dtensor : Subclass for *dense* tensors.
    sptensor : Subclass for *sparse* tensors.
    """

    shape = None
    __metaclass__ = ABCMeta

    def dimsize(self, ind):
        """ returns the size of the specified dimension.
        Same as shape[ind]."""
        return self.shape[ind]

    def ndims(self):
        """
        Number of dimensions of this tensor
        """
        return len(self.shape)

    def size(self):
        """returns the number of elements in the tensor"""
        return np.product(np.array(self.shape))

    @abstractmethod
    def copy(self):
        """
        Create a deep copy of the object
        """
        pass

    @abstractmethod
    def permute(self, order):
        """
        returns a tensor permuted by the order specified.
        """
        pass

    @abstractmethod
    def norm(self):
        """
        Frobenius norm of the tensor.
        """
        pass

    @abstractmethod
    def mttkrp(self, U, n):
        """
        Matricized tensor times Khatri-Rao product for tensor.
        Calculates the matrix product of the n-mode matricization of X with
        the Khatri-Rao product of all entries in U except the nth.

        Parameters
        ----------
        U - factorization

        Returns
        -------
        out : Khatri-Rao product as a numpy array
        """
        pass

    @abstractmethod
    def ttm(self, mat, dims=None, option=None):
        """
        Computes the tensor times the given matrix.
        arrs is a single 2-D matrix/array or a list of those matrices/arrays.
        """
        pass

    @abstractmethod
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
        out : a tensor
        """
        pass

    @abstractmethod
    def tondarray(self):
        """return an ndarray that contains the data of the tensor"""
        pass

    @abstractmethod
    def save(self, filename):
        pass
