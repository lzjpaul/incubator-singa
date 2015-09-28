import numpy as np
from scipy import spatial

import khatrirao
from dtensor import dtensor
import tools
import tensorTools as tt


def _binarizeFactors(A):
    A.normalize_sort(1)
    binA = [(A.U[n] != 0) for n in range(len(A.U))]
    return binA


def _calc_Bin_Congruence(A, B, R, dist='hamming'):
    binA = _binarizeFactors(A)
    binB = _binarizeFactors(B)
    N = len(binA)
    rawC = [spatial.distance.cdist(binA[n].T, binB[n].T, dist)
            for n in range(N)]
    arrayC = np.asarray(rawC)
    arrayC = np.ones(arrayC.shape) - arrayC
    C = np.prod(arrayC, axis=0)
    return rawC, C


class ktensor(object):
    '''
    A tensor stored as a decomposed Kruskal operator.
    The code is the python implementation of the @ktensor folder in the
    MATLAB Tensor Toolbox
    '''
    shape = None
    lmbda = None
    U = None
    R = 0

    def __init__(self, lmbda, U):
        """
        The tensor object stored as a Kruskal operator (decomposed).
        X = sum_r [lambda_r outer(a_r, b_r, c_r)].
        The columns of matrices A,B,C are the associated a_r, b_r, c_r
        """
        self.lmbda = lmbda
        self.R = len(lmbda)
        self.U = U
        self.shape = [len(self.U[r]) for r in range(len(U))]
        self.shape = tuple(self.shape)

    def __str__(self):
        ret = "Kruskal decomposition tensor with size {0}\n".format(self.shape)
        ret += "Lambda: {0}\n".format(self.lmbda)
        for i in range(len(self.U)):
            ret += "U[{0}] = {1}\n".format(i, self.U[i])
        return ret

    def fixsigns(self):
        """
        For each vector in each factor, the largest magnitude entries of K
        are positive provided that the sign on pairs of vectors in a rank-1
        component can be flipped.
        """
        for r in range(self.R):
            sgn = np.zeros(self.ndims())
            for n in range(self.ndims()):
                idx = np.argmax(np.abs(self.U[n][:, r]))
                sgn[n] = np.sign(self.U[n][idx, r])
            negidx = np.nonzero(sgn == -1)[0]
            nflip = 2 * np.floor(len(negidx) / 2)

            for i in np.arange(nflip):
                n = negidx[i]
                self.U[n][:, r] = -self.U[n][:, r]
        return

    def innerprod(self, Y):
        """
        Compute the inner product between this tensor and Y.
        If Y is a ktensor, the inner product is computed using inner products
        of the factor matrices.
        Otherwise, the inner product is computed using ttv with all of the
        columns of X's factor matrices
        """
        res = 0
        if isinstance(Y, ktensor):
            M = np.outer(self.lmbda, Y.lmbda)
            for n in range(self.ndims()):
                M = np.multiply(M, np.inner(self.U[n], Y.U[n]))
            res = np.sum(M)
        else:
            vecs = [{} for i in range(self.ndims())]
            for r in range(self.R):
                for n in range(self.ndims()):
                    vecs[n] = self.U[n][:, r]
                res = res + self.lmbda[r] * Y.ttv(vecs, range(self.ndims()))
        return res

    def ndims(self):
        return len(self.U)

    def norm(self):
        """ returns the Frobenius norm of the tensor."""
        coefMatrix = np.outer(self.lmbda, self.lmbda)
        for i in range(self.ndims()):
            coefMatrix = np.multiply(coefMatrix,
                                     np.dot(self.U[i].T, self.U[i]))
        return np.sqrt(np.abs(np.sum(coefMatrix)))

    def normalize(self, normtype=2):
        """"
        Normalize the column of each factor matrix U where the excess weight
        is absorbed by lambda. Also ensure lamda is positive.
        """
        # Normalize the matrices
        for n in range(self.ndims()):
            self.normalize_mode(n, normtype)
        idx = np.count_nonzero(self.lmbda < 0)
        if idx > 0:
            for i in np.nonzero(self.lmbda < 0):
                self.U[0][:, i] = -1 * self.U[0][:, i]
                self.lmbda[i] = -1 * self.lmbda[i]

    def normalize_mode(self, mode, normtype):
        """Normalize the ith factor using the norm specified by normtype"""
        colNorm = np.apply_along_axis(np.linalg.norm, 0, self.U[mode],
                                      normtype)
        zeroNorm = np.where(colNorm == 0)[0]
        colNorm[zeroNorm] = 1
        self.lmbda = self.lmbda * colNorm
        self.U[mode] = self.U[mode] / colNorm[np.newaxis, :]

    def normalize_sort(self, normtype=2):
        """"Normalize the column of each factor and
        sort each component/rank by magnitude greatest->smallest"""
        self.normalize(normtype)
        self.sort_components()

    def normalize_absorb(self, mode, normtype):
        """
        Normalize all the matrices using the norm specified by normtype and
        then absorb all the lambda magnitudes into the factors.
        """
        self.normalize(normtype)
        self.redistribute(mode)

    def permute(self, order):
        """
        Rearranges the dimensions of the ktensor so the order is
        specified by the vector order.
        """
        return ktensor(self.lmbda, self.U[order])

    def redistribute(self, mode):
        """
        Distribute the lambda values to a specified mode.
        Lambda vector is set to all ones, and the mode n takes on the values
        """
        self.U[mode] = self.U[mode] * self.lmbda[np.newaxis, :]
        self.lmbda = np.ones(self.R)

    """ Scoring Functions """
    def __calculateCongruences(self, B):
        # first make sure both are normalized
        self.normalize(2)
        B.normalize(2)
        C, rawC = tt.calc_tensor_sim(self.U, B.U)
        rawP = tt.calc_weight_sim(self.lmbda, B.lmbda)
        C = rawP * C
        return rawC, rawP, C

    @staticmethod
    def _greedy_index(C, R):
        selfR = []
        BR = []
        score = []
        for r in range(R):
            maxIdx = np.unravel_index(C.argmax(), C.shape)
            selfR.append(maxIdx[0])
            BR.append(maxIdx[1])
            score.append(C[maxIdx])
            C[maxIdx[0], :] = 0
            C[:, maxIdx[1]] = 0
        return selfR, BR, score

    def greedy_fms(self, B):
        """
        Compute the factor match score based on greedy search of the
        permutations. So the best matching factor first, then next, etc.
        """
        rawC, rawP, C = self.__calculateCongruences(B)
        selfR, BR, score = ktensor._greedy_index(C, B.R)
        sc = {'OrigOrder': selfR, 'OtherOrder': BR,
              'Lambda': rawP[selfR, BR].tolist(),
              'Score': score}
        selfR = np.array(selfR)
        BR = np.array(BR)
        for n in range(self.ndims()):
            sc[str(n)] = (rawC[n][selfR, BR]).tolist()
        return sc

    def fms(self, B):
        """
        Compute the factor match score based on the best possible permutation
        Use Munkres algorithm (Hungarian algorithm) to find the optimal path
        """
        rawC, rawP, C = self.__calculateCongruences(B)
        selfR, BR, indexes = tt.opt_index(C)
        sc = {'Order': indexes,
              'Lambda': rawP[selfR, BR].tolist(),
              'Score': C[selfR, BR].tolist()}
        for n in range(self.ndims()):
            sc[str(n)] = (rawC[n][selfR, BR]).tolist()
        return sc

    def greedy_fos(self, B, dist="hamming"):
        """
        Compute the factor over score based on greedy search of permutations
        So start with the first factor and work your way downards
        """
        rawC, C = _calc_Bin_Congruence(self, B, self.R, dist)
        selfR, BR, score = ktensor._greedy_index(C, B.R)
        sc = {'OrigOrder': selfR, 'OtherOrder': BR, 'Type': dist}
        selfR = np.array(selfR)
        BR = np.array(BR)
        for n in range(self.ndims()):
            sc[str(n)] = (rawC[n][selfR, BR]).tolist()
        return sc

    def fos(self, B, dist="hamming"):
        """ Factor overlap score """
        rawC, C = _calc_Bin_Congruence(self, B, self.R, dist)
        selfR, BR, indexes = tt.opt_index(C)
        sc = {'Order': indexes,
              'Type': dist,
              'Score': C[selfR, BR].tolist()}
        for n in range(self.ndims()):
            sc[str(n)] = (rawC[n][selfR, BR]).tolist()
        return sc

    def sort_components(self):
        """ Sort the ktensor components by magnitude, greatest to least."""
        sortidx = np.argsort(self.lmbda)[::-1]
        self.lmbda = self.lmbda[sortidx]
        # resort the u's
        for i in range(self.ndims()):
            self.U[i] = self.U[i][:, sortidx]

    def to_dtensor(self):
        """Convert this to a dense tensor"""
        tmp = khatrirao.khatrirao_array(self.U, True)
        data = np.inner(self.lmbda, tmp)
        return dtensor(data, self.shape)

    def ttv(self, v, dims):
        """
        Computes the product of the Kruskal tensor with the column vector along
        specified dimensions.

        Parameters
        ----------
        v - column vector
        dims - dimensions to multiply the product

        Returns
        -------
        out :
        """
        (dims, vidx) = tools.tt_dimscheck(dims, self.ndims(), len(v))
        remdims = np.setdiff1d(range(self.ndims()), dims)
        # Collapse dimensions that are being multiplied out
        newlmbda = self.lmbda
        for i in range(self.ndims()):
            newlmbda = np.inner(np.inner(self.U[dims[i]], v[vidx[i]]))
        if len(remdims) == 0:
            return np.sum(newlmbda)
        return ktensor(newlmbda, self.u[remdims])

    def copy(self):
        """Create a deep copy of the tensor"""
        return ktensor(np.copy(self.lmbda),
                       [self.U[n].copy() for n in range(len(self.U))])

    def save(self, filename):
        # store all the stuff in raw format
        outfile = file(filename, "wb")
        np.save(outfile, self.lmbda)
        np.save(outfile, self.ndims())
        [np.save(outfile, self.U[n]) for n in range(self.ndims())]
        outfile.close()

    def _check_object(self, other):
        if not isinstance(other, ktensor):
            raise NotImplementedError("Can only handle same ktensors object")
        if self.shape != other.shape:
            raise ValueError("Shapes of the tensors do not match")

    # Mathematic and Logic functions
    def __add__(self, other):
        self._check_object(other)
        lambda_ = self.lmbda + other.lmbda
        U = [np.concatenate((self.U[m], other.U[m]),
                            axis=1) for m in range(self.ndims())]
        return ktensor(lambda_, U)

    def __sub__(self, other):
        self._check_object(other)
        lambda_ = np.append(self.lmbda, -other.lmbda)
        U = [np.concatenate((self.U[m], other.U[m]),
                            axis=1) for m in range(self.ndims())]
        return ktensor(lambda_, U)

    def __eq__(self, other):
        if other is None:
            return False
        self._check_object(other)
        if self.lmbda != other.lmbda:
            return False
        # if lambda is true, then continue onto the other components
        for m in range(self.ndims()):
            if np.min(np.equal(self.U[m], other.U[m])):
                return False
        return True

    @staticmethod
    def load(filename):
        """ Load the tensor from a file """
        infile = open(filename, "rb")
        lmbda = np.load(infile)
        N = np.load(infile)
        U = []
        for n in range(N):
            U.append(np.load(infile))
        return ktensor(lmbda, U)
