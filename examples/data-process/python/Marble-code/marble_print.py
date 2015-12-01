import numpy as np
from abc import ABCMeta, abstractmethod

from ktensor import ktensor
import tensorTools as tt

AUG_LOCATION = 1
REG_LOCATION = 0
AUG_MIN = 1e-100

_DEF_MAXITER = 500
_DEF_MINIT = None

__all__ = ["Marble", "AUG_LOCATION", "REG_LOCATION", "_DEF_MAXITER",
           "_DEF_MINIT", "project_u", "project_A",
           "compute_xi", "calculate_ll"]


def _save_ktensor(outfile, M):
    np.save(outfile, M.lmbda)
    [np.save(outfile, factMat) for factMat in M.U]


def project_u(U):
    U, _ = tt.l1_normalize(U.clip(AUG_MIN))
    return U


def project_A(A, gamma):
    """
    Project the signal factor A onto the feasible space
    Basically hard-thresholding on the gamma
    """
    A, _ = tt.l1_normalize(A)
    # hard threshold
    A = tt.hardThresholdMatrix(A, gamma)
    # do a quick sanity check
    badCol = np.where(np.sum(A, axis=0) == 0)
    A[:, badCol] = 1
    A, _ = tt.l1_normalize(A)
    return A


def compute_xi(lastLL, currentLL, xi):
    xiDenom = np.max(np.absolute([lastLL, currentLL]))
    xiTemp = 1 - np.min([1, (np.absolute(lastLL - currentLL))]) / xiDenom
    if xiTemp > xi:
        # take the mean of the two
        xi = (xi + xiTemp) / 2
    return xi


def calculate_ll(X, M):
    """
    Computes the log-likelihood of the Poission regression
    Specifically, ll = -(sum_i m_i - x_i * log_i) where i is a
    multiindex across all tensor dimensions

    Parameters
    ----------
    X : input tensor of the class tensor or sptensor
    M : list of tensor decomposition (note that M can be single or many)

    Returns
    -------
    out : log likelihood value
    """
    N = X.ndims()
    MHat = M
    # check if it's still broken into two ktensors - if so combine
    if isinstance(M, dict):
        lHat = np.array([]).reshape(1, 0)
        AHat = [np.array([]).reshape(X.shape[n], 0) for n in range(N)]
        for v in M.values():
            lHat = np.append(lHat, v.lmbda)
            for n in range(N):
                print "AHat shape = \n", AHat[n].shape
                print "v.U[n] shape = \n", v.U[n].shape
            AHat = [np.column_stack((AHat[n], v.U[n])) for n in range(N)]
        MHat = ktensor(lHat, AHat)
    return tt.gen_kl_fit(X, MHat)


class Marble(object):
    obs_tensor = None   # Observed tensor
    dim_num = 0         # Number of dimensions
    cp_rank = 0         # CP decomposition rank
    cp_decomp = None    # The CP decomposition
    converg_tol = 0     # Convergence tolerance
    max_iterations = 0  # Maximum number of iterations
    __metaclass__ = ABCMeta

    def initialize(self, M=None):
        """
        Initialize the tensor decomposition
        """
        if M is None:
            AU = tt.random_factors(self.obs_tensor.shape, 1)
            F = tt.random_factors(self.obs_tensor.shape,
                                  self.cp_rank)
            self.cp_decomp = {
                REG_LOCATION: ktensor(np.ones(self.cp_rank), F),
                AUG_LOCATION: ktensor(np.ones(1), AU)
            }
        else:
            # do a quick sanity check
            if len(M) != 2:
                raise ValueError("Initialization needs to be of size 2")
            if not isinstance(M[AUG_LOCATION], ktensor):
                raise ValueError("Augmented location not ktensor type")
            if not isinstance(M[REG_LOCATION], ktensor):
                raise ValueError("Regular location not ktensor type")
            self.cp_decomp = M

    @abstractmethod
    def compute_decomp(self, **kwargs):
        pass

    @abstractmethod
    def project_data(self, xHat, n, **kwargs):
        pass

    def adjust_aug_factors(self, n):
        self.cp_decomp[AUG_LOCATION].U[n] =\
            project_u(self.cp_decomp[AUG_LOCATION].U[n])

    def save(self, filename):
        outfile = open(filename, "wb")
        np.save(outfile, self.dim_num)
        _save_ktensor(outfile, self.cp_decomp[REG_LOCATION])
        _save_ktensor(outfile, self.cp_decomp[AUG_LOCATION])
        outfile.close()

    def compare_factors(self, TM):
        return TM.greedy_fms(self.cp_decomp[REG_LOCATION])

    @staticmethod
    def _load_ktensor(infile, N):
        lambda_ = np.load(infile)
        U = [np.load(infile) for n in range(N)]
        return ktensor(lambda_, U)

    @abstractmethod
    def get_signal_factors(self):
        pass

    @staticmethod
    def load_decomp(filename):
        with open(filename, "rb") as infile:
            N = np.load(infile)
            M = {}
            M[REG_LOCATION] = Marble._load_ktensor(infile, N)
            M[AUG_LOCATION] = Marble._load_ktensor(infile, N)
        return M
