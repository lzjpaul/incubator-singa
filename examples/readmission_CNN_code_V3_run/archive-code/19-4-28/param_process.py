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

    def calculate_clip_coefficient(self, conv1_grad_list, dense_grad_list, threshold, step):
        clip_coefficient_dict = {}
        ### conv1
        conv1_grad_concatenate = conv1_grad_list[0].reshape(1,-1)
        for i in range(1,len(conv1_grad_list)):
            conv1_grad_concatenate = np.concatenate((conv1_grad_concatenate, conv1_grad_list[i].reshape(1,-1)), axis=1)
        if step % 200 == 0:
            print 'conv1_grad_concatenate shape: ', conv1_grad_concatenate.shape
            print 'conv1_grad_concatenate norm: ', np.linalg.norm(conv1_grad_concatenate)
        clip_coefficient_dict['conv1'] = threshold / (np.linalg.norm(conv1_grad_concatenate) + 1e-6)
        ### dense
        dense_grad_concatenate = dense_grad_list[0].reshape(1,-1)
        for i in range(1,len(dense_grad_list)):
            dense_grad_concatenate = np.concatenate((dense_grad_concatenate, dense_grad_list[i].reshape(1,-1)), axis=1)
        if step % 200 == 0:
            print 'dense_grad_concatenate shape: ', dense_grad_concatenate.shape
            print 'dense_grad_concatenate norm: ', np.linalg.norm(dense_grad_concatenate)
        clip_coefficient_dict['dense'] = threshold / (np.linalg.norm(dense_grad_concatenate) + 1e-6)
        return clip_coefficient_dict
    
    def regularization(self, coefficient, value, grad):
        if coefficient != 0:
            tensor.axpy(coefficient, value, grad)
        return grad

    def constraint(self, name, grad, clip_coefficient_dict):
        # print 'name: ', name
        # print 'clip_coefficient_dict: ', clip_coefficient_dict
        # print 'before clip grad l2: ', grad.l2() 
        if 'conv1' in name:
            clip_coefficient = clip_coefficient_dict['conv1']
        else:
            clip_coefficient = clip_coefficient_dict['dense']
        # print 'clip coefficient: ', clip_coefficient
        if clip_coefficient < 1.0:
            grad *= clip_coefficient
        # print 'after clip grad l2: ', grad.l2()
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

    def apply_with_regularizer_constraint_noise(self, coefficient, name, value, grad, clip_coefficient_dict, dev):
        """apply regularization and constraint if available.


        Returns:
            the updated gradient Tensor
        """
        # print 'in apply_with_regularizer_constraint_noise'
        # regularization
        grad = self.regularization(coefficient, value, grad)

        # constraint
        grad = self.constraint(name, grad, clip_coefficient_dict)

        # add noise
        grad = self.add_noise(grad, dev)

        return grad
