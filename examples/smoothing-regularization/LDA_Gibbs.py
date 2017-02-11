"""
(C) Zhaojing, Shaofeng, Jinyang - 2017

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation
"""
# overall logic (self.gaussians[w] = pi) -- step by step
# nk is a vector!!!!
# inf potential danger (power)?
import numpy as np
import scipy as sp
import sys
from scipy.special import gammaln

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    # print "in sample index"
    # print "sample_index p: ", p
    return np.random.multinomial(1,p).argmax()

def log_multi(alpha, K=None): # not checked!!!
    """
    Logarithm of the multinomial.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(np.log(alpha)) - np.log(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * np.log(alpha) - np.log(K*alpha)

class LdaSampler(object):

    def __init__(self, n_gaussians, alpha=0.1, a=1, b=2):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_gaussians = n_gaussians
        self.a = a
        self.b = b
        self.alpha = alpha
        print "self.n_gaussians: ", self.n_gaussians
        print "self.a: ", self.a
        print "self.b: ", self.b
        print "self.alpha: ", self.alpha

    def _initialize(self, weight_vec):
        # number of times gaussian k is sampled
        self.nk = np.zeros(self.n_gaussians)
        # variance of the weight for each gaussian
        self.nbk = np.zeros(self.n_gaussians)
        for k in range(self.nbk.shape[0]):
            self.nbk[k] = self.b
        self.gaussians = {} # which gaussian this weight dimension refers to
        self.pre_weight_vec = np.zeros(weight_vec.shape[0])
        self.gibbsbatchiter = 0

        for w in range(weight_vec.shape[0]):
            pi = np.random.randint(self.n_gaussians)
            # print "pi: ", pi
            self.nk[pi] += 1
            self.nbk[pi] += (weight_vec[w] * weight_vec[w] * 0.5)
            self.gaussians[w] = pi
            self.pre_weight_vec[w] = weight_vec[w]
        # print "self.nk: ", self.nk
        # print "self.nbk: ", self.nbk
        # print "self.gaussians: ", self.gaussians

    def _conditional_distribution(self, w, weight_vec):
        """
        Conditional distribution (vector of size n_topics).
        """
        # left = np.zeros(self.nk.shape[0])
        # for k in range(self.nk.shape[0]):
        #    left[k] = self._weight_cond_pi(w, k, weight_vec)
        left_term1 = np.power((1 - (0.5 * weight_vec[w] * weight_vec[w] / (self.nbk + 0.5 * weight_vec[w] * weight_vec[w]))), (self.a + self.nk / 2.0))
        if np.sum(left_term1) == float('Inf'):
            print "term1 overflow"
            exit(1)
        left_term2 = 1.0 / np.power((self.nbk + 0.5 * weight_vec[w] * weight_vec[w]), 0.5)
        left_term3 = np.exp(gammaln(0.5 + self.a + self.nk / 2.0) - gammaln(self.a + self.nk / 2.0))
        left = (left_term1 * left_term2 * left_term3)
        # print "left: ", left
        right = (self.nk + self.alpha)
        # print "slef.nk: ", self.nk
        # print "self.alpha: ", self.alpha
        # print "right: ", right
        p_pi = left * right
        # print "p_pi: ", p_pi
        # normalize to obtain probabilities
        p_pi /= np.sum(p_pi)
        # print "p_pi norm: ", p_pi
        # print "p_pi shape: ", p_pi.shape
        return p_pi

#    def loglikelihood(self): # not checked!!!
#        """
#        Compute the likelihood that the model generated the data.
#        """
#        lik = 0
#        # not checked !!!
#        for k in xrange(self.n_gaussians):
#            lik += (self.nk[k] / 2.0) * (-1) * np.log(2 * np.pi)
#            lik += np.log(self.a + (self.nk[k] / 2.0))
#            lik -= (self.a + (self.nk[k] / 2.0)) * np.log(self.nbk[k])
#
#        lik += log_multi(self.nk+self.alpha)
#        lik -= log_multi(self.alpha, self.n_gaussians)
#
#        return lik

    def run(self, weight_vec, num_iter, batchgibbs, maxiter=1, eps=0.0001):
        """
        Run the Gibbs sampler.
        """
        print "num_iter: ", num_iter
        if num_iter == 0:
            print "num_iter: ", num_iter
            print "initialization"
            self._initialize(weight_vec)

        # joint_liklihood = 0
        if batchgibbs == 0:
            w_range_begin = 0
            w_range_end = weight_vec.shape[0]
        else:
            w_range_begin = ((batchgibbs * self.gibbsbatchiter) % weight_vec.shape[0])
            if (w_range_begin + batchgibbs) < weight_vec.shape[0]:
                w_range_end = (w_range_begin + batchgibbs)
                self.gibbsbatchiter = self.gibbsbatchiter + 1
            else:
                w_range_end = weight_vec.shape[0]
                self.gibbsbatchiter = 0 # new epoc of features


        if batchgibbs != 0:
            print "w_range_begin: ", w_range_begin
            print "w_range_end: ", w_range_end
            print "self.gibbsbatchiter: ", self.gibbsbatchiter
        for it in xrange(maxiter):
            # print "iteration: ", it
            for w in range(w_range_begin, w_range_end):
                # if w % 50000 == 0 and w != 0:
                #     print "w: ", w
                # print "w: ", w
                self.nk[self.gaussians[w]] -= 1
                self.nbk[self.gaussians[w]] -= (self.pre_weight_vec[w] * self.pre_weight_vec[w] * 0.5)
                # print "minus self.nk: ", self.nk
                # print "minus self.nbk: ", self.nbk
                # print "minus self.gaussians: ", self.gaussians

                p_pi = self._conditional_distribution(w, weight_vec)
                pi = sample_index(p_pi)
                # print "pi: ", pi

                self.nk[pi] += 1
                self.nbk[pi] += (weight_vec[w] * weight_vec[w] * 0.5)
                self.gaussians[w] = pi # !!!
                self.pre_weight_vec[w] = weight_vec[w]
                #if pi == 0:
                #    print "nk == 1 w: ", w
                #    print "nk == 1 weight_vec[w]: ", weight_vec[w]
                # print "weight_vec[w]: ", weight_vec[w]
                # print "pi: ", pi
                # print "self.nbk[4]: ", self.nbk[4]
                # print "self.nbk: ", self.nbk
                # print "add self.nk: ", self.nk
                # print "add self.nbk: ", self.nbk
                # print "add self.gaussians: ", self.gaussians


            #if it == 0:
            #    joint_liklihood = self.loglikelihood() # not checked
            #else:
            #    cur_joint_liklihood = self.loglikelihood() # not checked
            #    if np.absolute(cur_joint_liklihood -joint_liklihood) < eps:
            #        break
            #    joint_liklihood = cur_joint_liklihood


        theta_vec = (self.nk + self.alpha) / (weight_vec.shape[0] + self.nk.shape[0]*self.alpha)
        # lambda_vec = np.zeros(self.nk.shape[0])
        # for k in range (lambda_vec.shape[0]):
        #    lambda_vec[k] = (self.a + 0.5 * self.nk[k]) / self.nbk[k]
        print "LDA self.nbk: ", self.nbk
        print "LDA self.nk: ", self.nk
        lambda_vec = (self.a + 0.5 * self.nk) / self.nbk
        print "LDA lambda_vec: ", lambda_vec

        return theta_vec, lambda_vec

if __name__ == "__main__":
    np.random.seed(10)
    n_gaussians = int(sys.argv[1])
    weight_vec = np.random.rand(15) #15
    print "weight_vec: ", weight_vec
    print "weight_vec.shape[0]: ", weight_vec.shape[0] #one-dimension array
    sampler = LdaSampler(n_gaussians, alpha = 0.5, a = 2, b = 5) #number of gaussians
    theta_vec, lambda_vec = sampler.run(weight_vec)
    print "theta_vec: ", theta_vec
    print "sum(theta_vec): ", sum(theta_vec)
    print "lambda_vec: ", lambda_vec
    # print "Likelihood", sampler.loglikelihood()
