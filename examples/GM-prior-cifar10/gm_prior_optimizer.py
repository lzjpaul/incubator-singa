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

class GMOptimizer(Optimizer):
    '''
    introduce hyper-parameters for GM-regularization: a, b, alpha
    '''
    def __init__(self, cpudev=None, net=None, hyperpara=None, gm_num=None, pi=None, reg_lambda=None, 
                 lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        Optimizer.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)
        self.gmregularizer = GMRegularizer(hyperpara=hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda)
        self.weight_dim_list = []
        self.weight_name_list = []
        for (s, p) in zip(net.param_names(), net.param_values()):
            print "param name: ", s
            print "param shape: ", tensor.to_numpy(p).shape
            if np.ndim(tensor.to_numpy(p)) == 2:
                self.weight_name_list.append(s)
                dims = tensor.to_numpy(p).shape[0] * tensor.to_numpy(p).shape[1]
                print "dims: ", dims
                self.weight_dim_list.append(dims)
        self.weightdimSum = sum(self.weight_dim_list)
        print "self.weightdimSum: ", self.weightdimSum
        print map(lambda x: x.encode('ascii'), self.weight_name_list) 
        print "self.weight_dim_list: ", self.weight_dim_list

    def apply_GM_regularizer_constraint(self, dev, cpudev, trainnum, net, weight_name_list, weight_dim_list, weightdimSum, epoch, value, grad, name, step):
        # if np.ndim(tensor.to_numpy(value)) <= 2:
        if np.ndim(tensor.to_numpy(value)) != 2: 
            self.apply_regularizer_constraint(epoch, value, grad, name, step)
        else: # weight parameter
            grad = self.gmregularizer.apply(dev, cpudev, trainnum, net, weight_name_list, weight_dim_list, weightdimSum, 
                                            self.weight_name_list.index(name)==0, self.weight_name_list.index(name)==(len(self.weight_name_list)-1),
                                            epoch, value, grad, name, step)
        return grad


class GMRegularizer(Regularizer):
    '''GM regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2)
    '''

    def __init__(self, hyperpara=None, gm_num=None, pi=None, reg_lambda=None):
        self.a, self.b, self.alpha, self.gm_num = hyperpara[0], hyperpara[1], hyperpara[2], gm_num
        print "self.a, self.b, self.alpha, self.gm_num: ", self.a, self.b, self.alpha, self.gm_num
        self.pi, self.reg_lambda = np.reshape(np.array(pi), (1, gm_num)), np.reshape(np.array(reg_lambda), (1, gm_num))
        print "init self.reg_lambda: ", self.reg_lambda
        print "init self.pi: ", self.pi

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        # responsibility normalized with pi
        responsibility = gaussian.pdf(self.w_array, loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        # responsibility normalized with summation(denominator)
        self.responsibility = responsibility/(np.sum(responsibility, axis=1).reshape(self.w_array.shape))
    
    def update_GM_Prior_EM(self, weightdimSum, epoch):
        # update pi
        self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (2 * self.b + np.sum(self.responsibility * np.square(self.w_array), axis=0))
        if epoch < 2:
            print "np.sum(self.responsibility, axis=0): ", np.sum(self.responsibility, axis=0)
            print "np.sum(self.responsibility * np.square(self.w[:-1]), axis=0): ", np.sum(self.responsibility * np.square(self.w_array), axis=0)
        # update reg_lambda
        self.pi = (np.sum(self.responsibility, axis=0) + self.alpha - 1) / (weightdimSum + self.gm_num * (self.alpha - 1))
        if epoch < 2:
            print 'reg_lambda', self.reg_lambda
            print 'pi:', self.pi

    # extract weights from all kinds of params (bias, ...,)
    def extract_weight(self, cpudev, net):
        # extract w array
        self.w_array = None
        for (s, p) in zip(net.param_names(), net.param_values()):
            if np.ndim(tensor.to_numpy(p)) == 2:
                if self.w_array is None:
                    self.w_array = tensor.to_numpy(p).reshape((1, -1))
                else:
                    self.w_array = np.concatenate((self.w_array, tensor.to_numpy(p).reshape((1, -1))), axis=1)
        self.w_array = self.w_array.reshape((-1, 1))

    def apply(self, dev, cpudev, trainnum, net, weight_name_list, weight_dim_list, weightdimSum, isfirst, islast, epoch, value, grad, name, step):
        # new iteration: update responsibility & extract weight array & calculate reg_grad_w
        if isfirst: 
            self.extract_weight(cpudev, net) # usef for calculating responsibility and reg_grad_w
            self.calcResponsibility()
            self.reg_grad_w = np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w_array.shape) * self.w_array
            self.weight_dim_index = 0
        # calculate the grad for the specific weight
        this_weight_dim = weight_dim_list[weight_name_list.index(name)] 
        grad = tensor.from_numpy((self.reg_grad_w[self.weight_dim_index:(self.weight_dim_index+this_weight_dim)].reshape(tensor.to_numpy(value).shape[0], -1))/float(trainnum))
        grad.to_device(dev)
        self.weight_dim_index = (self.weight_dim_index + this_weight_dim) ## why previously param_dim_0 * param_dim_1, correct ???
        # update pi and lambda
        if islast:
            self.update_GM_Prior_EM(weightdimSum, epoch)
        return grad

class GMSGD(GMOptimizer, SGD):
    '''The vallina Stochasitc Gradient Descent algorithm with momentum.
    But this SGD has a GM regularizer
    '''

    def __init__(self, cpudev=None, net=None, hyperpara=None, gm_num=None, pi=None, reg_lambda=None, 
                 lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        GMOptimizer.__init__(self, cpudev=cpudev, net=net, hyperpara=hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda, 
                                  lr=lr, momentum=momentum, weight_decay=weight_decay, regularizer=regularizer,
                                  constraint=constraint)
        SGD.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)
        # conf = model_pb2.OptimizerConf()
        # if self.momentum is not None:
        #     conf.momentum = self.momentum
        # conf.type = 'sgd'
        # self.opt = singa.CreateOptimizer('SGD')
        # self.opt.Setup(conf.SerializeToString())
        

    # compared with apply_with_lr, this need one more argument: isweight
    def apply_with_lr(self, dev, cpudev, trainnum, net, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value
        ##### GM-prior: using gm_regularizer ##############
        grad = self.apply_GM_regularizer_constraint(dev=dev, cpudev=cpudev, trainnum=trainnum, net=net, weight_name_list=self.weight_name_list, weight_dim_list=self.weight_dim_list, weightdimSum=self.weightdimSum, epoch=epoch, value=value, grad=grad, name=name, step=step)
        ##### GM-prior: using gm_regularizer ##############
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value
