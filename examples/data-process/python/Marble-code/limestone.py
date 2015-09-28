import numpy as np

import CP_APR
import ktensor
import tensorTools


class Limestone(object):
    obs_tensor = None
    cp_rank = 0
    orig_decomp = None
    sparse_decomp = None

    def __init__(self, X, R):
        self.obs_tensor = X
        self.cp_rank = R

    def decompose(self, **kwargs):
        threshold = kwargs.pop('thr', np.repeat(1e-4, self.obs_tensor.ndims()))
        self.orig_decomp, itstats = CP_APR.cp_apr(self.obs_tensor,
                                                  self.cp_rank, **kwargs)
        self.orig_decomp.normalize_sort(1)
        self.sparse_decomp = self.orig_decomp.copy()
        print "obs_tensor shape = \n", self.obs_tensor.shape
        for n in range(len(self.obs_tensor.shape)):
            self.sparse_decomp.U[n] = \
                tensorTools.hardThresholdMatrix(self.sparse_decomp.U[n],
                                                threshold[n])
        return itstats

    def projectData(self, X, n, **kwargs):
        """
        Project a slice, solving for the factors of the nth mode

        Parameters
        ------------
        X : the tensor to project onto the basis
        n : the mode to project onto
        iters : the max number of inner iterations
        epsilon : parameter to avoid dividing by zero
        convTol : the convergence tolerance

        Output
        -----------
        the projection matrix
        """
        iters = kwargs.pop('iters', 10)
        epsilon = kwargs.pop('epsilon', 1e-10)
        convTol = kwargs.pop('convTol', 1e-4)
        # Setup the 'initial guess'
        F = []
        for m in range(X.ndims()):
            if m == n:
                F.append(np.random.rand(X.shape[m], self.cp_rank))
            else:
                # double check the shape is the right dimensions
                if (self.sparse_decomp.U[m].shape[0] != X.shape[m]):
                    raise ValueError("Shape of the tensor X is incorrect")
                F.append(self.sparse_decomp.U[m])
        M = ktensor.ktensor(np.ones(self.cp_rank), F)
        # Solve for the subproblem
        M, Phi, totIter, kktMV = CP_APR.solveForModeB(X, M, n, iters, epsilon,
                                                      convTol)
        return tensorTools.norm_rows(M.U[n])
