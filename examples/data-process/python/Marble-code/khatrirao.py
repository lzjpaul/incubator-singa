'''
Compute Khatri-Rao product of matrices
'''
import numpy as np


def khatrirao(A, B):
    """
    Computes the Khatri-Rao product of Matrices A and B

    Parameters
    ----------
    A, B = numpy arrays (2-D) only

    Returns
    -------
    out : Khatri-Rao product as a numpy array
    """
    # Make sure the columns are same numbers
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matrices must have the same number of columns")
    N = A.shape[1]
    M = A.shape[0] * B.shape[0]
    # Pre-allocate the memory
    P = np.zeros((M, N))
    for n in range(N):
        ab = np.outer(A[:, n], B[:, n])
        P[:, n] = ab.flatten()
    return P


def khatrirao_array(U, reverse=False):
    """
    Computes the Khatri-Rao product of a list of matrices

    Parameters
    ----------
    U = list of 2-D numpy arrays
    reverse = reverse the order of computation

    Returns
    -------
    out : Khatri-Rao product as a numpy array
    """
    tmpU = U
    if reverse:
        tmpU = U[::-1]
    return reduce(khatrirao, tmpU)
