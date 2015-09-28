'''
Compute the nonnegative tensor factorization using alternating Poisson
regression
'''
from collections import OrderedDict
import numpy as np
import time

from ktensor import ktensor
import tensorTools

_DEF_MAXITER = 1000
_DEF_MAXINNER = 10
_DEF_CONVTOL = 1e-4
_DEF_EPSILON = 1e-10
_DEF_KAPPA = 1e-2
_DEF_KAPPATOL = 1e-10
_DEF_MINIT = None


def solveForModeB(X, M, n, maxInner, epsilon, tol):
    """
    Solve for the subproblem B = argmin (B >= 0) f(M)

    Parameters
    ----------
    X : the original tensor
    M : the current CP factorization
    n : the mode that we are trying to solve the subproblem for
    epsilon : parameter to avoid dividing by zero
    tol : the convergence tolerance

    Returns
    -------
    M : the updated CP factorization
    Phi : the last Phi(n) value
    iter : the number of iterations before convergence / maximum
    kktModeViolation : the maximum value of min(B(n), E - Phi(n))
    """
    # Pi(n) = [A(N) kr A(N-1) kr ... A(n+1) kr A(n-1) kr .. A(1)]^T
    Pi = tensorTools.calculatePi(X, M, n)
    for iter in range(maxInner):
        # Phi = (X(n) elem-div (B Pi)) Pi^T
        Phi = tensorTools.calculatePhi(X, M.U[n], Pi, n, epsilon=epsilon)
        # check for convergence that min(B(n), E - Phi(n)) = 0 [or close]
        kktModeViolation = np.max(np.abs(np.minimum(M.U[n],
                                                    1 - Phi).flatten()))
        if (kktModeViolation < tol):
            break
        # Do the multiplicative update
        M.U[n] = np.multiply(M.U[n], Phi)
    return M, Phi, iter, kktModeViolation


def _solveSubproblem(X, M, n, maxInner, epsilon, tol):
    """ """
    # Shift the weight from lambda to mode n
    # B = A(n)*Lambda
    M.redistribute(n)
    # solve the inner problem
    M, Phi, i, kktModeViolation = solveForModeB(X, M, n, maxInner,
                                                epsilon, tol)
    # Shift weight from mode n back to lambda
    M.normalize_mode(n, 1)
    return M, Phi, i, kktModeViolation, i == 0


def cp_apr(X, R, **kwargs):
    """
    Compute nonnegative CP with alternative Poisson regression.
    Code is the python implementation of cp_apr in the MATLAB Tensor Toolbox

    Parameters
    ----------
    X : input tensor of the class tensor or sptensor
    R : the rank of the CP
    Minit : the initial guess (in the form of a ktensor), if None random guess
    tol : tolerance on the inner KKT violation
    maxiters : maximum number of iterations
    maxinner : maximum number of inner iterations
    epsilon : parameter to avoid dividing by zero
    kappatol : tolerance on complementary slackness
    kappa : offset to fix complementary slackness

    Returns
    -------
    M : the CP model as a ktensor
    cpStats: the statistics for each inner iteration
    modelStats: a dictionary item with the final statistics for the
                factorization
    """
    M = kwargs.pop('init', _DEF_MINIT)
    maxinner = kwargs.pop('max_inner', _DEF_MAXINNER)
    maxiters = kwargs.pop('max_iter', _DEF_MAXITER)
    kappatol = kwargs.pop('kappa_tol', _DEF_KAPPATOL)
    tol = kwargs.pop('tol', _DEF_CONVTOL)
    epsilon = kwargs.pop('epsilon', _DEF_EPSILON)
    kappa = kwargs.pop('kappa', _DEF_KAPPA)
    debug = "Iteration {0} with elapsed time={1}"
    N = X.ndims()

    # Random initialization
    if M is None:
        M = ktensor(np.ones(R), tensorTools.random_factors(X.shape, R))
    M.normalize(1)
    Phi = [[] for i in range(N)]
    is_conv = [True for i in range(N)]
    # statistics
    iter_info = OrderedDict(sorted({}.items(), key=lambda t: t[1]))
    for iteration in range(maxiters):
        startIter = time.time()
        for n in range(N):
            info = {}
            nViolations = 0
            startMode = time.time()
            # Make adjustments to M[n] entries that violate
            # complementary slackness
            if iteration > 0:
                V = np.logical_and(Phi[n] > 1, M.U[n] < kappatol)
                if np.count_nonzero(V) > 0:
                    nViolations = nViolations + 1
                    M.U[n][V > 0] = M.U[n][V > 0] + kappa
            M, Phi[n], inner, kktMV, is_conv[n] = _solveSubproblem(X, M, n,
                                                                   maxinner,
                                                                   epsilon,
                                                                   tol)
            elapsed = time.time() - startMode
            info[str(n)] = {"inner": inner, "kkt": kktMV, "time": elapsed,
                            "lsqr": tensorTools.lsqr_fit(X, M),
                            "ll": tensorTools.gen_kl_fit(X, M)}
        elapsed = time.time() - startIter
        info["violation"] = nViolations
        iter_info[str(iteration)] = info
        print(debug.format(iteration, elapsed))
        if all(is_conv):
            break

    # Print the statistics
    fit = tensorTools.lsqr_fit(X, M)
    ll = tensorTools.gen_kl_fit(X, M)
    print("Number of iterations = {0}".format(iteration))
    print("Final least squares fit = {0}".format(fit))
    print("Final log-likelihood = {0}".format(ll))
    iter_info['Final'] = {"lsqr": fit, "ll": ll}
    return M, iter_info
