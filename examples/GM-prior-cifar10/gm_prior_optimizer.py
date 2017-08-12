# This file contains all the classes that are related to 
# GM-prior adaptive regularizer.
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
import random
class GMOptimizer(Optimizer):
    '''
    introduce hyper-parameters for GM-regularization: a, b, alpha
    '''
    def __init__(self, net=None, hyperpara=None, gm_num=None, pi=None, reg_lambda=None, uptfreq=None, 
                 lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        Optimizer.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)
        # self.gmregularizer = GMRegularizer(hyperpara=hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda)
        self.weight_name_list = {}
        self.weight_dim_list = {}
        self.gmregularizers = {}
        self.extract_layer_name_dim_gmregularizer(net, hyperpara=hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda, uptfreq=uptfreq)

    def layer_wise_hyperpara(self, fea_num):
        print "layer_wise fea_num: ", fea_num
        b, alpha = [(0.3 * fea_num), (0.5 * fea_num), (0.7 * fea_num), (0.9 * fea_num), (fea_num), (3 * fea_num), (5 * fea_num), (7 * fea_num), (9 * fea_num), (0.3 * fea_num * 1e-1), (0.5 * fea_num * 1e-1), (0.7 * fea_num * 1e-1), (0.9 * fea_num * 1e-1), (fea_num * 1e-1),\
                   (fea_num * 0.3 * 1e-2), (0.5 * fea_num * 1e-2), (0.7 * fea_num * 1e-2), (0.9 * fea_num * 1e-2), (fea_num * 1e-2), (0.3 * fea_num * 1e-3), (0.5 * fea_num * 1e-3), (0.7 * fea_num * 1e-3), (0.9 * fea_num * 1e-3), (fea_num * 1e-3)],\
                   [fea_num**(0.9), fea_num**(0.7), fea_num**(0.5), fea_num**(0.3)]
        alpha_val = random.choice(alpha)
        b_val = random.choice(b)
        a = [(1. + b_val * 1e-1), (1. + b_val * 1e-2)]
        a_val = random.choice(a)
        return [a_val, b_val, alpha_val]

    def extract_layer_name_dim_gmregularizer(self, net, hyperpara, gm_num, pi, reg_lambda, uptfreq):
        for (s, p) in zip(net.param_names(), net.param_values()):
            print "param name: ", s
            print "param shape: ", tensor.to_numpy(p).shape
            if np.ndim(tensor.to_numpy(p)) == 2:
                self.weight_name_list[s] = s
                dims = tensor.to_numpy(p).shape[0] * tensor.to_numpy(p).shape[1]
                print "dims: ", dims
                self.weight_dim_list[s] = dims
                layer_hyperpara = self.layer_wise_hyperpara(dims) # layerwise initialization of hyper-params
                self.gmregularizers[s] = GMRegularizer(hyperpara=layer_hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda, uptfreq=uptfreq)
        self.weightdimSum = sum(self.weight_dim_list.values())
        print "self.weightdimSum: ", self.weightdimSum
        print "self.weight_name_list: ", self.weight_name_list
        print "self.weight_dim_list: ", self.weight_dim_list


    def apply_GM_regularizer_constraint(self, dev, trainnum, net, epoch, value, grad, name, step):
        # if np.ndim(tensor.to_numpy(value)) <= 2:
        if np.ndim(tensor.to_numpy(value)) != 2:
            return self.apply_regularizer_constraint(epoch, value, grad, name, step)
        else: # weight parameter
            print "run time name: ", name
            grad = self.gmregularizers[name].apply(dev, trainnum, net, epoch, value, grad, name, step)
            return grad


class GMRegularizer(Regularizer):
    '''GM regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''

    def __init__(self, hyperpara=None, gm_num=None, pi=None, reg_lambda=None, uptfreq=None):
        self.a, self.b, self.alpha, self.gm_num = hyperpara[0], hyperpara[1], hyperpara[2], gm_num
        print "init self.a, self.b, self.alpha, self.gm_num: ", self.a, self.b, self.alpha, self.gm_num
        self.pi, self.reg_lambda = np.reshape(np.array(pi), (1, gm_num)), np.reshape(np.array(reg_lambda), (1, gm_num))
        print "init self.reg_lambda: ", self.reg_lambda
        print "init self.pi: ", self.pi
        self.gmuptfreq, self.paramuptfreq = uptfreq[0], uptfreq[1]
        print "self.gmuptfreq, self.paramuptfreq: ", self.gmuptfreq, self.paramuptfreq

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        # responsibility normalized with pi
        responsibility = gaussian.pdf(self.w_array, loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        # responsibility normalized with summation(denominator)
        self.responsibility = responsibility/(np.sum(responsibility, axis=1).reshape(self.w_array.shape))
    
    def update_GM_Prior_EM(self, epoch):
        # update pi
        self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (2 * self.b + np.sum(self.responsibility * np.square(self.w_array), axis=0))
        # print "np.sum(self.responsibility, axis=0): ", np.sum(self.responsibility, axis=0)
        # print "np.sum(self.responsibility * np.square(self.w[:-1]), axis=0): ", np.sum(self.responsibility * np.square(self.w_array), axis=0)
        # update reg_lambda
        self.pi = (np.sum(self.responsibility, axis=0) + self.alpha - 1) / (self.w_array.shape[0] + self.gm_num * (self.alpha - 1))
        # print 'reg_lambda', self.reg_lambda
        # print 'pi:', self.pi
        # print 'self.w_array.shape[0]: ', self.w_array.shape[0]

    def apply(self, dev, trainnum, net, epoch, value, grad, name, step):
        print "runtime self.a, self.b, self.alpha, self.gm_num: ", self.a, self.b, self.alpha, self.gm_num
        self.w_array = tensor.to_numpy(value).reshape((-1, 1)) # used for EM update also
        if epoch < 2 or step % self.paramuptfreq == 0:
            self.calcResponsibility()
            self.reg_grad_w = np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w_array.shape) * self.w_array
        reg_grad_w_dev = tensor.from_numpy((self.reg_grad_w.reshape(tensor.to_numpy(value).shape[0], -1))/float(trainnum))
        reg_grad_w_dev.to_device(dev)
        grad.to_device(dev)
        tensor.axpy(1.0, reg_grad_w_dev, grad)
        if epoch < 2 or step % self.gmuptfreq == 0:
            if epoch >=2 and step % self.paramuptfreq != 0:
                self.calcResponsibility()
            self.update_GM_Prior_EM(epoch)
        return grad

class GMSGD(GMOptimizer, SGD):
    '''The vallina Stochasitc Gradient Descent algorithm with momentum.
    But this SGD has a GM regularizer
    '''

    def __init__(self, net=None, hyperpara=None, gm_num=None, pi=None, reg_lambda=None, uptfreq=None, 
                 lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        GMOptimizer.__init__(self, net=net, hyperpara=hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda, uptfreq=uptfreq, 
                                  lr=lr, momentum=momentum, weight_decay=weight_decay, regularizer=regularizer,
                                  constraint=constraint)
        SGD.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)

    # compared with apply_with_lr, this need one more argument: isweight
    def apply_with_lr(self, dev, trainnum, net, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value
        ##### GM-prior: using gm_regularizer ##############
        grad = self.apply_GM_regularizer_constraint(dev=dev, trainnum=trainnum, net=net, epoch=epoch, value=value, grad=grad, name=name, step=step)
        ##### GM-prior: using gm_regularizer ##############
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value
