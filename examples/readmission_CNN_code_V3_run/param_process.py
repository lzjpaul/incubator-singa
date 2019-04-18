from singa import tensor
import numpy as np
import pdb

class param_process(object):

    def __init__(self, mean, variance):
        
        # actaully, it is std
        self.mean, self.var = mean, variance


    def generateGaussianParam(self):
        """return mean and variance for gaussian"""
        return 0.0, 1.0

    def regularization(self, coefficient, value, grad):
        if coefficient != 0:
            tensor.axpy(coefficient, value, grad)
        return grad

    def constraint(self, grad, threshold):
        nrm = grad.l2()
        if threshold < nrm:
            grad *= (threshold/nrm)
        return grad

    def noise_function():
        pass

    def add_noise(self, grad, dev):
        # grad tensor to numpy
        # print 'in add_noise'
        noises = np.random.normal(self.mean, self.var, size=grad.shape)
        # print 'noises: ', noises
        noises = noises.astype(np.float32)
        noises_dev = tensor.Tensor(noises.shape, dev)
        noises_dev.copy_from_numpy(noises)
        # print 'tensor.to_numpy(noises_dev): ', tensor.to_numpy(noises_dev)
        tensor.axpy(1.0, noises_dev, grad)
        return grad

    def apply_with_regularizer_constraint_noise(self, coefficient, value, grad, threshold, dev):
        """apply regularization and constraint if available.


        Returns:
            the updated gradient Tensor
        """
        # print 'in apply_with_regularizer_constraint_noise'
        # regularization
        grad = self.regularization(coefficient, value, grad)

        # constraint
        grad = self.constraint(grad, threshold)

        # add noise
        grad = self.add_noise(grad, dev)

        return grad
