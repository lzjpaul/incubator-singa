"""
File that contains common methods for tensor methods
"""
import numpy as np
from sklearn import preprocessing
from munkres import Munkres
import sys
from numba import jit

import accumarray
from sptensor import sptensor
from dtensor import dtensor
import khatrirao
import tenmat


def random_factors(modeDim, R):
    """
    Randomly initialize column normalized matrices
    """
    tmp = [np.random.rand(m, R) for m in modeDim]
    return [preprocessing.normalize(fm, axis=0, norm='l1') for fm in tmp]


def uniform_factors(modeDim, R):
    """
    Initialize the matrix using a uniform distribution
    """
    tmp = [np.ones((m, R)) for m in modeDim]
    return [preprocessing.normalize(fm, axis=0, norm='l1') for fm in tmp]


def count_nnz(U, axis):
    """
    Count the number of non-zero elements for each axis
    """
    return (np.apply_along_axis(np.count_nonzero, axis, U)).tolist()


def count_ktensor_nnz(M, axis=0):
    """
    Count the number of non-zero elements for each mode in the ktensor
    """
    return [count_nnz(M.U[mode], axis) for mode in range(M.ndims())]


def int_overflow(val):
    return np.float(val)


def get_norm(v, normtype):
    """
    Get the norm of a vector
    """
    val = np.linalg.norm(v, ord=normtype)
    if val == 0:
        val = 1
    if not np.isfinite(val):
        print val
    # fix overflow
    val = int_overflow(val)
    return val


def col_normalize(matrix, normtype=1):
    colNorm = np.apply_along_axis(get_norm, 0, matrix, normtype)
    return matrix / colNorm[np.newaxis, :]


def l1_normalize(matrix):
    """
    Stochastic normalization
    """
    colNorm = matrix.sum(axis=0)
    # deal with zero norm case
    zeroNorm = np.where(colNorm == 0)[0]
    colNorm[zeroNorm] = 1
    matrix = matrix / colNorm[np.newaxis, :]
    return matrix, colNorm


def calc_cosine_sim(A, B):
    """
    Calculate the similarity between two matrices
    """
    return abs(np.dot(A.transpose(), B))


def calc_tensor_sim(orig_A, new_A):
    rawC = [calc_cosine_sim(orig_A[n], new_A[n])
            for n in range(len(orig_A))]
    return np.prod(np.asarray(rawC), axis=0), rawC


def calc_weight_sim(orig_L, new_L):
    R = len(orig_L)
    la = np.tile(orig_L, (R, 1))
    lb = np.tile(new_L, (R, 1)).T
    min_entries = 1e-10 * np.ones((la.shape))
    diff = np.abs(la - lb) / np.maximum(np.maximum(la, lb),
                                        min_entries)
    return np.ones((R, R)) - diff


def opt_index(C):
    copyC = np.ones(C.shape) - C.copy()
    hAlg = Munkres()
    indexes = hAlg.compute(copyC)
    rowIdx, colIdx = map(np.array, zip(*indexes))
    return rowIdx, colIdx, indexes


def hardThresholdMatrix(U, thresh):
    """
    Perform hard thresholding of a matrix

    Parameters
    ------------
    U : the matrix to threshold
    thresh : the threshold value (anything below is chopped to zero)

    Output
    -----------
    U: the new thresholded matrix
    """
    # do a quick sanity check
    if not np.all(np.isfinite(U)):
        raise ValueError("Error in U" + str(U))
    zeroIdx = np.where(U < thresh)
    U[zeroIdx] = 0
    return U


def range_omit_k(N, k):
    """
    Return the range of numbers from 0 -> N but omitting k
    """
    return [i for i in range(N) if i != k]


def _row_norm(row):
    """
    Normalize such that elements sum to 1
    """
    weight = np.sum(row)
    if weight < 1e-50:
        # create a vector of ones
        row = np.ones(row.shape)
        weight = np.sum(row)
    return row / weight


def norm_rows(M):
    """
    Apply the normalize of a row along the axis
    """
    return np.apply_along_axis(_row_norm, 1, M)


def calculatePi(X, M, n):
    """
    Calculate the product of all matrices but the n-th
    (Eq 3.6 in Chi + Kolda ArXiv paper)
    """
    Pi = None
    modes_but_n = range_omit_k(X.ndims(), n)
    if isinstance(X, sptensor):
        Pi = np.ones((X.nnz(), M.R))
        for nn in modes_but_n:
            Pi = np.multiply(M.U[nn][X.subs[:, nn], :], Pi)
    else:
        Pi = khatrirao.khatrirao_array([M.U[i] for i in modes_but_n],
                                       reverse=True)
    return Pi


def calculatePhi(X, B, Pi, n, epsilon=1e-4, C=None):
    """
    Calculate the matrix for multiplicative update

    Parameters
    ----------
    X       : the observed tensor
    B       : the factor matrix associated with mode n
    Pi      : the product of all matrices but the n-th from above
    n       : the mode that we are trying to solve the subproblem for
    epsilon : the
    C       : the augmented / non-augmented tensor (\alpha u \Psi or B \Phi) in
              sparse form
    """
    Phi = None
    if isinstance(X, sptensor):
        Phi = -np.ones((X.shape[n], B.shape[1]))
        xsubs = X.subs[:, n]
        if C is not None:
            v = np.sum(np.multiply(B[xsubs, :], Pi) + C, axis=1)
        else:
            v = np.sum(np.multiply(B[xsubs, :], Pi), axis=1)
        wvals = X.vals.flatten() / v
        for r in range(B.shape[1]):
            Phi[:, r] = accumarray.accum_np(xsubs,
                                            np.multiply(wvals, Pi[:, r]),
                                            size=X.shape[n])
    else:
        Xn = tenmat.tenmat(X, [n])
        V = np.inner(B, Pi)
        W = Xn.data / np.maximum(V, epsilon)
        Phi = np.inner(W, Pi.transpose())
    return Phi


def lsqr_fit(X, M):
    """
    Calculate the fraction of the residual explained by the factorization
    Parameters
    ------------
    X : observed tensor
    M : factorized tensor
    """
    normX = X.norm()
    normresidual = np.sqrt(np.square(normX) + np.square(M.norm()) -
                           2 * M.innerprod(X))
    fit = 1 - (normresidual / normX)
    return fit


def _calc_sptensor_kl(X, M, N):
    xsubs = X.subs
    # absorb lambda into mode
    C = (M.U[0] * M.lmbda[np.newaxis, :])[xsubs[:, 0], :]
    for n in range(1, N):
        C = np.multiply(C, M.U[n][xsubs[:, n], :])
    mhat_vals = np.sum(C, axis=1)
    ll = np.sum(M.U[0] * M.lmbda[np.newaxis, :]) - \
        np.sum(np.multiply(X.vals.flatten(),
                           np.log(mhat_vals)))
    return ll


@jit(nopython=True)
def _calc_dtensor_kl(Xdata, Mdata):
    return np.sum(Mdata - Xdata * np.log(Mdata))


def gen_kl_fit(X, M):
    """
    Computes the log-likelihood of model M given data X.
    Specifically, ll = -(sum_i m_i - x_i * log_i) where i is a
    multiindex across all tensor dimensions

    Parameters
    ----------
    X : input tensor of the class tensor or sptensor
    M : list of tensor decompositiosn (note that M can be single or many)

    Returns
    -------
    out : log likelihood value
    """
    if isinstance(X, sptensor):
        return _calc_sptensor_kl(X, M, X.ndims())
    else:
        MHat = M
        if not isinstance(M, dtensor):
            MHat = M.to_dtensor()
        return _calc_dtensor_kl(X._data.flatten(), MHat._data.flatten())
