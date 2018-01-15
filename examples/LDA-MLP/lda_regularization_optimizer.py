# This file contains all the classes that are related to 
# LDA-prior adaptive regularizer.
# =============================================================================
from singa import optimizer
from singa.optimizer import SGD
from singa.optimizer import Regularizer
from singa.optimizer import Optimizer
import numpy as np
from singa import tensor
from singa import singa_wrap as singa
from singa.proto import model_pb2
from scipy.stats import norm as gaussian
import math

class LDAOptimizer(Optimizer):
    '''
    introduce hyper-parameters for LDA-regularization: alpha
    '''
    def __init__(self, net=None, lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        Optimizer.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)

    def lda_register(self, hyperpara, ldapara, phi, uptfreq):
        theta = [1.0/ldapara[1] for _ in range(ldapara[1])]
        self.ldaregularizer = LDARegularizer(hyperpara=hyperpara, ldapara=ldapara, theta=theta, phi=phi, uptfreq=uptfreq)

    def apply_LDA_regularizer_constraint(self, dev, trainnum, net, epoch, value, grad, name, step):
        # if np.ndim(tensor.to_numpy(value)) <= 2:
        if name != 'dense1/weight':
            return self.apply_regularizer_constraint(epoch, value, grad, name, step)
        else: # ip1/weight parameter
            grad = self.ldaregularizer.apply(dev, trainnum, net, epoch, value, grad, name, step)
            return grad


class LDARegularizer(Regularizer):
    '''LDA regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''

    def __init__(self, hyperpara=None, ldapara=None, theta=None, phi=None, uptfreq=None):
        self.alpha, self.phi = hyperpara[0], np.copy(phi)
        self.doc_num, self.topic_num, self.word_num = ldapara[0], ldapara[1], ldapara[2]
        print "self.alpha, self.doc_num, self.topic_num, self.word_num: ", self.alpha, self.doc_num, self.topic_num, self.word_num
        print "self.phi shape: ", self.phi.shape
        self.theta_alldoc = np.zeros((self.doc_num, self.topic_num))
        for doc_idx in range(self.doc_num):
            self.theta_alldoc[doc_idx,:] = np.copy(theta)
        print "init self.theta_alldoc: ", self.theta_alldoc
        self.ldauptfreq, self.paramuptfreq = uptfreq[0], uptfreq[1]
        print "init self.ldauptfreq, self.paramuptfreq: ", self.ldauptfreq, self.paramuptfreq

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        self.responsibility_all_doc = np.zeros((self.doc_num, self.word_num, self.topic_num))
        for doc_idx in range(self.doc_num):
            responsibility_doc = self.phi*(self.theta_alldoc[doc_idx].reshape((1, -1)))
            # responsibility normalized with summation(denominator)
            self.responsibility_all_doc[doc_idx] = responsibility_doc/(np.sum(responsibility_doc, axis=1).reshape(-1,1))

    def calcRegGrad(self):
        theta_phi_all_doc = np.zeros((self.word_num, self.doc_num))
        for doc_idx in range(self.doc_num):
            theta_phi_doc = self.phi*(self.theta_alldoc[doc_idx].reshape((1, -1)))
            theta_phi_doc = np.sum(theta_phi_doc, axis=1)
            theta_phi_doc = np.log(theta_phi_doc) 
            theta_phi_all_doc[:, doc_idx] = theta_phi_doc
        return -(np.sign(self.w_array) * theta_phi_all_doc)
    
    def update_LDA_EM(self, name, step):
        self.theta_alldoc = np.zeros((self.doc_num, self.topic_num))
        # update theta_all_doc
        for doc_idx in range(self.doc_num):
            theta_doc = (np.sum((self.responsibility_all_doc[doc_idx] * np.absolute(self.w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (self.alpha - 1)) / np.sum(np.sum((self.responsibility_all_doc[doc_idx] * np.absolute(self.w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (self.alpha - 1))
            self.theta_alldoc[doc_idx] = theta_doc
            if step % self.ldauptfreq == 0:
                print 'theta_alldoc:', self.theta_alldoc

    def apply(self, dev, trainnum, net, epoch, value, grad, name, step):
        self.w_array = tensor.to_numpy(value)
        if epoch < 2 or step % self.paramuptfreq == 0:
            self.calcResponsibility()
            # self.reg_grad_w = np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w_array.shape) * self.w_array
            self.reg_grad_w = self.calcRegGrad()
        reg_grad_w_dev = tensor.from_numpy(self.reg_grad_w/float(trainnum))
        reg_grad_w_dev.to_device(dev)
        grad.to_device(dev)
        if (epoch == 0 and step < 50) or step % self.ldauptfreq == 0:
            print "step: ", step
            print "name: ", name
            print "data grad l2 norm: ", grad.l2()
            print "reg_grad_w_dev l2 norm: ", reg_grad_w_dev.l2()
        tensor.axpy(1.0, reg_grad_w_dev, grad)
        if (epoch == 0 and step < 50) or step % self.ldauptfreq == 0:
            print "delta w norm: ", grad.l2()
            print "w norm: ", value.l2()
        if epoch < 2 or step % self.ldauptfreq == 0:
            if epoch >=2 and step % self.paramuptfreq != 0:
                self.calcResponsibility()
            self.update_LDA_EM(name, step)
        return grad

class LDASGD(LDAOptimizer, SGD):
    '''The vallina Stochasitc Gradient Descent algorithm with momentum.
    But this SGD has an LDA regularizer
    '''

    def __init__(self, net=None, lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        LDAOptimizer.__init__(self, net=net, lr=lr, momentum=momentum, weight_decay=weight_decay, regularizer=regularizer,
                                  constraint=constraint)
        SGD.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)

    # compared with apply_with_lr, this need one more argument: isweight
    def apply_with_lr(self, dev, trainnum, net, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value
        ##### LDA-prior: using lda_regularizer ##############
        grad = self.apply_LDA_regularizer_constraint(dev=dev, trainnum=trainnum, net=net, epoch=epoch, value=value, grad=grad, name=name, step=step)
        ##### LDA-prior: using lda_regularizer ##############
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value