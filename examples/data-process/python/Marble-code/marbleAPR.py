'''
Compute the nonnegative tensor factorization using
alternating Poisson regression

This is the algorithm described in the paper Marble.
'''
import numpy as np
import time
from collections import OrderedDict

from marble import *       # NOQA
import tensorTools

_DEF_MAXINNER = 10
_DEF_GRADUAL = True
_DEF_DELTATOL = 1e-2


def project_A(A, gamma):
    """
    Project the signal factor A onto the feasible space
    Basically hard-thresholding on the gamma
    """
    # first make sure they are above zero
    A = np.maximum(A, np.zeros(A.shape))
    # if they're all the same just return uniform
    if len(np.unique(A)) == 1:
        A = np.ones(A.shape)
        A, _ = tensorTools.l1_normalize(A)
        return A
    # normalize
    A, _ = tensorTools.l1_normalize(A)
    # hard threshold
    A = tensorTools.hardThresholdMatrix(A, gamma)
    # do a quick sanity check
    badCol = np.where(np.sum(A, axis=0) == 0)
    A[:, badCol] = 1
    A, _ = tensorTools.l1_normalize(A)
    return A


class MarbleAPR(Marble):
    alpha = 0

    def __init__(self, X, R, alpha):
        self.obs_tensor = X
        self.dim_num = X.ndims()
        self.cp_rank = R
        self.alpha = alpha

    def normalizeAugTensor(self):
        """
        Normalize the augmented tensor to the value alpha
        """
        self.cp_decomp[AUG_LOCATION].normalize(1)
        self.cp_decomp[AUG_LOCATION].lmbda = np.repeat(self.alpha, 1)

    def _solveMode(self, Pi, B, C, n, maxInnerIters, convTol=1e-4):
        """
        Performs the inner iterations and checks for convergence.
        """
        for innerI in range(maxInnerIters):
            # Phi = (X(n) elem-div (B Pi)) Pi^T
            Phi = tensorTools.calculatePhi(self.obs_tensor, B, Pi, n, C=C)
            # check for convergence that min(B(n), E - Phi(n)) = 0 [or close]
            kktModeViolation = np.max(np.abs(np.minimum(B, 1 - Phi).flatten()))
            if (kktModeViolation < convTol):
                break
            # Do the multiplicative update
            B = np.multiply(B, Phi)
        return B, (innerI + 1), kktModeViolation

    def solveSubproblem(self, C, aug, n, maxInnerIters):
        """
        Solve the subproblem for mode n

        Parameters
        ------------
        C : the "other" tensor, either bias or signal tensor
        aug : the location to solve (either augmented or not)
        n : the mode to solve
        """
        # shift the weight from lambda to mode n
        self.cp_decomp[aug].redistribute(n)
        Pi = tensorTools.calculatePi(self.obs_tensor, self.cp_decomp[aug], n)
        B, inI, kktModeViolation = self._solveMode(Pi,
                                                   self.cp_decomp[aug].U[n],
                                                   C, n, maxInnerIters)
        self.cp_decomp[aug].U[n] = B
        # shift the weight from mode to lambda
        self.cp_decomp[aug].normalize_mode(n, 1)
        return B, Pi, inI, kktModeViolation

    def _solveAugmentedTensor(self, xsubs, B, Pi, n, maxInnerIters):
        # now that we are done, we can calculate the new
        # 'unaugmented matricization'
        Chat = np.multiply(B[xsubs, :], Pi)
        _, _, inI2, kktMV2 = self.solveSubproblem(Chat, AUG_LOCATION, n,
                                                  maxInnerIters)
        self.adjust_aug_factors(n)
        self.normalizeAugTensor()
        return inI2, kktMV2

    def _solveSignalTensor(self, xsubs, n, maxInnerIters):
        BHat = self.cp_decomp[AUG_LOCATION].U[n]
        Psi = tensorTools.calculatePi(self.obs_tensor,
                                      self.cp_decomp[AUG_LOCATION], n)
        C = np.multiply(BHat[xsubs, :], Psi)
        return self.solveSubproblem(C, REG_LOCATION, n, maxInnerIters)

    def _project_factor(self, n, thr, loc=REG_LOCATION):
        self.cp_decomp[loc].U[n] = project_A(self.cp_decomp[loc].U[n], thr)
        return np.dot(self.cp_decomp[loc].U[n],
                      np.diag(self.cp_decomp[loc].lmbda))

    def get_signal_factors(self):
        return self.cp_decomp[REG_LOCATION]

    def compute_decomp(self, **kwargs):
        # initialization options
        gamma = kwargs.pop('gamma', list(np.repeat(0, self.dim_num)))
        gradual = kwargs.pop('gradual', _DEF_GRADUAL)
        M = kwargs.pop('init', _DEF_MINIT)
        max_inner = kwargs.pop('max_inner', _DEF_MAXINNER)
        max_iters = kwargs.pop('max_iter', _DEF_MAXITER)
        delta_tol = kwargs.pop('del_tol', _DEF_DELTATOL)
        debug_log = "Iteration {0}: Xi = {1}, dll = {2}, time = {3}"
        print "delta_tol = \n", delta_tol
        print "gradual = \n", gradual
        print "max_inner = \n", max_inner
        print "max_iters = \n", max_iters
        print "gamma = \n", gamma

        print "begin initialize\n"
        self.initialize(M)
        # Dictionary to manage iteration information
        print "begin orderDict\n"
        iterInfo = OrderedDict(sorted({}.items(), key=lambda t: t[1]))
        print "begin calculate_ll\n"
        lastLL = calculate_ll(self.obs_tensor, self.cp_decomp)
        # projection factor starts at 0 (unless there's no gradual)
        xi = 0 if gradual else 1
        # if nothing is set, we're just not going to do any hard-thresholding
        # for outer iterations
        print "begin iterations\n"
        for iteration in range(max_iters):
            startIter = time.time()
            print "iteration = \n", iteration
            print "startIter = \n", startIter
            for n in range(self.dim_num):
                startMode = time.time()
                # first we calculate the "augmented" tensor matricization
                self.cp_decomp[AUG_LOCATION].redistribute(n)
                xsubs = self.obs_tensor.subs[:, n]
                B, Pi, inI1, kktMV1 = self._solveSignalTensor(xsubs, n,
                                                              max_inner)
                # hard threshold based on the xi and gamma
                thr = xi * gamma[n]
                if (thr > 0):
                    B = self._project_factor(n, thr)
                elapsed1 = time.time() - startMode
                # calculate the new 'unaugmented matricization'
                inI2, kktMV2 = self._solveAugmentedTensor(xsubs, B, Pi, n,
                                                          max_inner)
                elapsed2 = time.time() - startMode
                ll = calculate_ll(self.obs_tensor, self.cp_decomp)
                iterInfo[str((iteration, n))] = {
                    "Time": [elapsed1, elapsed2],
                    "KKTViolation": [kktMV1, kktMV2],
                    "Iterations": [inI1, inI2],
                    "LL": ll
                }
            if gradual:
                xi = compute_xi(lastLL, ll, xi)
            print "Iteration " + str(iteration) + ": " + str(ll)
            print(debug_log.format(iteration, xi, np.abs(lastLL - ll),
                                   time.time() - startIter))
            print "lastLL = \n", lastLL
            print "ll = \n", ll
            print "abs(lastLL - ll) = \n", np.abs(lastLL - ll)
            if np.abs(lastLL - ll) < delta_tol and xi >= 0.99:
                break
            lastLL = ll
        return iterInfo, ll

    def project_data(self, XHat, n, **kwargs):
        max_iters = kwargs.pop('max_iter', _DEF_MAXITER)
        max_inner = kwargs.pop('max_inner', _DEF_MAXINNER)
        delta_tol = kwargs.pop('delta_tol', _DEF_DELTATOL)

        print "max_iters = \n", max_iters
        print "max_inner = \n", max_inner
        print "delta_tol = \n", delta_tol
        # store off the old ones
        origM = {REG_LOCATION: self.cp_decomp[REG_LOCATION].copy(),
                 AUG_LOCATION: self.cp_decomp[AUG_LOCATION].copy()}
        origX = self.obs_tensor
        self.obs_tensor = XHat
        # randomize the nth
        aug_fact = np.random.rand(self.obs_tensor.shape[n], 1)
        self.cp_decomp[REG_LOCATION].lmbda = np.ones(self.cp_rank)
        self.cp_decomp[REG_LOCATION].U[n] = np.random.rand(self.obs_tensor.shape[n], self.cp_rank)
        self.cp_decomp[AUG_LOCATION].U[n] = aug_fact
        self.cp_decomp[AUG_LOCATION].lmbda = np.ones(1)
        # renormalize
        self.cp_decomp[REG_LOCATION].normalize(1)
        self.normalizeAugTensor()
        lastLL = calculate_ll(self.obs_tensor, self.cp_decomp)
        for iteration in range(max_iters):
            print "project_data iteration: = \n", iteration
            startIter_project = time.time()
            print "startIter = \n", startIter_project
            xsubs = self.obs_tensor.subs[:, n]
            B, Pi, _, _ = self._solveSignalTensor(xsubs, n, max_inner)
            self._solveAugmentedTensor(xsubs, B, Pi, n, max_inner)
            ll = calculate_ll(self.obs_tensor, self.cp_decomp)
            if np.abs(lastLL - ll) < delta_tol:
                break
            lastLL = ll
        # scale by summing across the rows
        projMat = tensorTools.norm_rows(self.cp_decomp[REG_LOCATION].U[n])
        biasMat = self.cp_decomp[AUG_LOCATION].U[n]
        self.cp_decomp = origM
        self.obs_tensor = origX
        return projMat, biasMat
