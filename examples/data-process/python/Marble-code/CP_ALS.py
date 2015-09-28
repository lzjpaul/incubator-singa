'''
Compute the CP decomposition using alternating least squares algorithm.
Code is the python implementation of cp_als in the MATLAB Tensor Toolbox.
'''
import numpy as np
from scipy import sparse

import ktensor
import tensorTools


def cp_als(X, R, tol=1e-4, maxiters=50):
    """
    Compute an estimate of the best rank-R CP model of a tensor X using an alternating
    least-squares algorithm. The fit is defined as 1 - norm(X - full(P))/norm(X) and is
    loosely the proportion of data described by the CP model.
    
    Parameters
    ----------
    X - input tensor of the class tensor or sptensor
    R - the rank of the CP

    Returns
    -------
    out : the CP model as a ktensor
    """
    N = X.ndims()      # number of dimensions
    U = [np.random.rand(X.shape[idx], R) for idx in range(N)]
    fit = 0
    
    for k in range(maxiters):
        fitold = fit
        # iterate over all the range
        for n in np.arange(N):
            # Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            Unew = X.mttkrp(U, n)
            # Compute the linear system coefficients
            Y = np.ones((R, R))
            for i in np.concatenate((np.arange(0, n), np.arange(n + 1, N))):
                Y = np.multiply(Y, np.dot(U[i].transpose(), U[i]))
            Unew = np.linalg.solve(Y, Unew.transpose()).transpose()
            
            # Normalize each vector to prevent singularities
            if k == 0:
                lambda_ = np.sqrt(np.sum(np.square(Unew), axis=0))
            else:
                lambda_ = Unew.max(axis=0)
            U[n] = Unew * sparse.spdiags(1 / lambda_, 0, R, R, format='csr')
        
        P = ktensor.ktensor(lambda_, U)
        fit = tensorTools.lsqr_fit(X, P)
        fitchange = abs(fitold - fit)
        print("Iteration {0}: fit={1} with delta={2}".format(k, fit, fitchange))
        if k > 0 and fitchange < tol:
            break
    
    ## Clean up the final result by normalizing the tensor
    P.normalize_sort()
    P.fixsigns()
    return P, fit
