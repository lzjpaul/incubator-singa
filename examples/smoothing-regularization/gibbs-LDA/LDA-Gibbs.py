"""
(C) Zhaojing, Shaofeng, Jinyang - 2017

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation
"""
# to line 51
# check: Tao + overall logic (self.gaussians[w] = pi)
# nk is a vector!!!!
# gamma_function merge
# inf potential danger?
# line 41
import numpy as np
import scipy as sp
import sys
# from scipy.special import gammaln

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def log_gamma(alpha):
    if alpha < 1:
        print "ak is too small (small than 0.5)"
        exit(1)
    if ((alpha - np.int(alpha)) == 0) # integer
        res = 0
        if alpha == 1:
            return res
        else: #alpha >= 2
            alpha = alpha - 1
            while alpha >= 1
                res += np.log(alpha)
                alpha = (alpha - 1)
    else: #1.5, 2.5, 3.5 ...
        alphan = (alpha - 0.5)
        res = 0
        for n in range((alphan + 1), (2 * alphan + 1)):
            res += np.log(n)
        res -= alphan * np.log(4)
        res += 0.5 * np.log(np.pi)


def log_multi(alpha, K=None):
    """
    Logarithm of the multinomial.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(np.log(alpha)) - np.log(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * np.log(alpha) - np.log(K*alpha)

def gamma_division(ak):
    if ak < 1:
        print "ak is too small (small than 0.5)"
        exit(1)

    if ((ak-np.int(ak)) == 0.5) # 2.5, 3.5, ...
        ak = np.int(ak) # the least is (1.5 - 0.5) == 1 ....
        product = 1.0 / np.power(np.pi, 0.5)
        t = (2 * ak - 1)
        while t >= 1:
            product = product * (t + 1) / t
            t = (t - 2)
        return product

    else:   # integer
        if ak == 1:
            return np.power(np.pi, 0.5) * 3 / 4
        else:
            product = 0.5 * np.power(np.pi, 0.5)
            t = (2 * ak - 2)
            while t >= 2:
                product = product * (t + 1) / t
                t = (t -2)
            return product

class LdaSampler(object):

    def __init__(self, n_gaussians, alpha=0.1, a, b):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_gaussians = n_gaussians
        self.a = a
        self.b = b

    def _initialize(self, weight_vec):
        # number of times gaussian k is sampled
        self.nk = np.zeros(self.n_gaussians)
        # variance of the weight for each gaussian
        self.nbk = np.zeros(self.n_gaussians)
        for k in range(nbk.shape[0])
            self.nbk[k] = self.b
        self.gaussians = {} # which gaussian this weight dimension refers to

        for w in range(weight_vec.shape[0]):
            pi = np.random.randint(self.n_gaussians)
            self.nk[pi] += 1
            self.nbk[pi] += (weight_vec[w] * weight_vec[w] * 0.5)
            self.gaussians[w] = pi

    def weight_cond_pi(w, k, weight_vec):
        term1 = np.power((1 - (0.5 * weight_vec[w] * weight_vec[w] / (self.nbk[k] + 0.5 * weight_vec[w] * weight_vec[w]))), (self.a + nk[k] / 2.0))
        term2 = 1.0 / np.power((self.nbk[k] + 0.5 * weight_vec[w] * weight_vec[w]), 0.5)
        term3 = gamma_division((self.a + nk[k] / 2.0))
        return (term1 * term2 * term3)

    def _conditional_distribution(self, w, weight_vec):
        """
        Conditional distribution (vector of size n_topics).
        """
        left = np.zeros(self.nk.shape[0])
        for k in range(self.nk.shape[0]):
            left[k] = weight_cond_pi(w, k, weight_vec)
        right = (self.nk + self.a)
        p_pi = left * right
        # normalize to obtain probabilities
        p_pi /= np.sum(p_pi)
        return p_pi

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        lik = 0

        for k in xrange(self.n_gaussians):
            lik += (nk[k] / 2.0) * (-1) * np.log(2 * np.pi)
            lik += log_gamma(self.a + (self.nk[k] / 2.0))
            lik -= (self.a + (self.nk[k] / 2.0)) * np.log(nbk[k])

        lik += log_multi(self.nk+self.a)
        lik -= log_multi(self.a, self.n_gaussians)

        return lik

    def run(self, weight_vec, maxiter=30, eps=0.0001):
        """
        Run the Gibbs sampler.
        """
        self._initialize(weight_vec)

        joint_liklihood = 0
        for it in xrange(maxiter):
            for w in range(weight_vec.shape[0]):
                self.nk[self.gaussians[w]] -= 1
                self.nbk[self.gaussians[w]] -= (weight_vec[w] * weight_vec[w] * 0.5)

                p_pi = self._conditional_distribution(w, weight_vec)
                pi = sample_index(p_pi)

                self.nk[pi] += 1
                self.nbk[pi] += (weight_vec[w] * weight_vec[w] * 0.5)
                self.gaussians[w] = pi # !!!

            if it == 0:
                joint_liklihood = loglikelihood()
            else:
                cur_joint_liklihood = loglikelihood()
                if np.absolute(cur_joint_liklihood -joint_liklihood) < eps:
                    break
                joint_liklihood = cur_joint_liklihood


        theta_vec = (self.nk + self.a) / (weight_vec.shape[0] + 1)
        lambda_vec = np.zeros(self.nk.shape[0])
        for k in range (lambda_vec.shape[0]):
            lambda_vec[k] = (self.a + 0.5 * nk[k]) / (self.b + 0.5 * nbk[k])

        return theta_vec, lambda_vec

if __name__ == "__main__":

    print "weight_vec.shape[0]: ", weight_vec.shape[0] #one-dimension array
    sampler = LdaSampler(n_gaussians, alpha = 1.0/(weight_vec.shape[0]), a = 2, b = 5) #number of gaussians
    theta_vec, lambda_vec = sampler.run(weight_vec)
    print "theta_vec: ", theta_vec
    print "lambda_vec: ", lambda_vec
    # print "Likelihood", sampler.loglikelihood()
