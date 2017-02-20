# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" CIFAR10 dataset is at https://www.cs.toronto.edu/~kriz/cifar.html.
It includes 5 binary dataset, each contains 10000 images. 1 row (1 image)
includes 1 label & 3072 pixels.  3072 pixels are 3 channels of a 32x32 image
"""

import cPickle
import numpy as np
import os
import argparse
from scipy import sparse
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2
from caffe import caffe_net

import alexnet
import vgg
import resnet


def load_dataset(filepath):
    print 'Loading data file %s' % filepath
    with open(filepath, 'rb') as fd:
        cifar10 = cPickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(dir_path, num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_test_data(dir_path):
    images, labels = load_dataset(dir_path + "/test_batch")
    return np.array(images,  dtype=np.float32), np.array(labels, dtype=np.int32)


def normalize_for_vgg(train_x, test_x):
    mean = train_x.mean()
    std = train_x.std()
    train_x -= mean
    test_x -= mean
    train_x /= std
    test_x /= std
    return train_x, test_x


def normalize_for_alexnet(train_x, test_x):
    mean = np.average(train_x, axis=0)
    train_x -= mean
    test_x -= mean
    return train_x, test_x


def vgg_lr(epoch):
    return 0.1 / float(1 << ((epoch / 25)))


def alexnet_lr(epoch):
    if epoch < 120:
        return 0.001
    elif epoch < 130:
        return 0.0001
    else:
        return 0.00001


def resnet_lr(epoch):
    if epoch < 81:
        return 0.1
    elif epoch < 122:
        return 0.01
    else:
        return 0.001


def caffe_lr(epoch):
    if epoch < 8:
        return 0.001
    else:
        return 0.0001

def gaussian_mixture_gd_descent_avg(res_matrix, w, theta_r_vec, theta_vec, lambda_t_vec, lambda_vec, a, b, theta_alpha, C):
    #if sparse.issparse(batch_X): batch_X = batch_X.toarray()
    # print "in lasso gd avg"
    w_array = np.copy(w) # in order for np.exp
    print "w_array shape: ", w_array.shape
    # print "w shape: ", w.shape
    # w_array = np.reshape(w_array, w_array.shape[1])
    w_weight_array = w_array.reshape((-1, 1)) # no bias
    # grad = param * w.sign()
    # print "lasso w shape: ", w.shape
    lambda_w = np.dot(w_weight_array, lambda_vec.reshape((1, -1)))
    print "lambda_w shape: ", lambda_w.shape
    grad = np.sum((res_matrix * lambda_w), axis=1)# -log(p(w))
    # print "grad[0:10]: ", grad[0:10]
    # print "(lambda_vec[0] * w_array)[0:10]: ", (lambda_vec[0] * w_array)[0:10]
    # grad = sparse.csr_matrix(grad)
    # print "grad shape: ", grad.shape

    ###lambda_t_update#########
    term1 = (float(a-1) / lambda_vec) - b
    res_w = sparse.csr_matrix(res_matrix.T).dot(sparse.diags(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])))
    res_w = (res_w.toarray().T)
    term2 = np.sum((res_matrix / (2.0 * lambda_vec.reshape(1, -1))) - res_w, axis=0)
    print "gd lambda_t term2: ", term2[:100]
    lambda_t_vec_update = (- term1 - term2) * lambda_vec # lambda = exp(lambda_t)derivative of lambda to t
    # lambda_t_vec_update = (- term1 - term2) * (-lambda_vec) # should change mapping as well!!! lambda^(-1) = exp(lambda_t)derivative of lambda to t
    ###lambda_t_update#########

    ###theta_r_update#########
    theta_k_r_j = np.zeros((theta_vec.shape[0], theta_vec.shape[0])) ## derivative of theta_k to r_j
    for k in range(theta_k_r_j[:,0].shape[0]):
        for j in range(theta_k_r_j[0, :].shape[0]):
            if k == j:
                theta_k_r_j[k, j] = (theta_vec[k] - theta_vec[k] * theta_vec[k])
            else:
                theta_k_r_j[k, j] = (- theta_vec[k] * theta_vec[j])

    theta_r_vec_update = np.zeros(theta_r_vec.shape[0])
    term1 = (theta_alpha - 1) / theta_vec.astype(float)
    print "theta_vec.astype(float): ", theta_vec.astype(float)
    term2 = np.sum(res_matrix.astype(float) / (theta_vec.reshape(1, -1)), axis=0)
    theta_derivative = ( - term1 - term2)
    for j in range(theta_r_vec_update.shape[0]): #r_j
        theta_r_vec_update[j] = np.sum(theta_derivative * theta_k_r_j[:, j])
    print "-term1: ", -term1
    print "-term2: ", -term2
    print "theta_derivative: ", theta_derivative
    print "theta_k_r_j: ", theta_k_r_j
    print "theta_r_vec_update: ", theta_r_vec_update


    ###theta_r_update#########
    return sparse.csr_matrix(grad), theta_r_vec_update, lambda_t_vec_update

def train(data, net, max_epoch, get_lr, weight_decay, n_gaussian=3, theta_alpha=300, a=1, b=1, C=1.0, batch_size=100, 
          use_cpu=False):
    print 'Start intialization............'
    if use_cpu:
        print 'Using CPU'
        dev = device.get_default_device()
        cpudev = dev
    else:
        print 'Using GPU'
        dev = device.create_cuda_gpu()
        cpudev = device.get_default_device()

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=0.0)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
    train_x, train_y, test_x, test_y = data
    num_train_batch = train_x.shape[0] / batch_size
    num_test_batch = test_x.shape[0] / batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    
    print "in train n_gaussian: ", n_gaussian
    print "in train theta_alpha: ", theta_alpha
    print "in train a: ", a
    print "in train b: ", b
    # initialization !!
    theta_r_vec = np.zeros(n_gaussian)
    theta_r_exp_vec = np.exp(theta_r_vec)
    theta_vec = theta_r_exp_vec / np.sum(theta_r_exp_vec)
    print "theta_vec initialization: ", theta_vec
    # lambda_t_vec = np.random.normal(0.0, 1, n_gaussian)
    lambda_t_vec = np.array([np.log(1/2.), np.log(1/4.), np.log(1/8.)])
    # lambda_t_vec = np.array([np.log(1/2.)])
    # lambda_vec = np.exp(lambda_t_vec)
    lambda_vec = np.exp(lambda_t_vec)

    for epoch in range(max_epoch):
        np.random.seed(epoch)
        np.random.shuffle(idx)
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            ##### calculate responsibility and weight dimension list#####
            weight_dim_list = []
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                if 'weight' in str(s):
                    p.to_device(cpudev)
                    dims = tensor.to_numpy(p).shape
                    weight_dim_list.append(dims)
                    if len(weight_dim_list) == 1:
                        w_array = tensor.to_numpy(p).reshape((1, -1))
                    else:
                        w_array = np.concatenate((w_array, tensor.to_numpy(p).reshape((1, -1))), axis=1)
                    p.to_device(dev)
            print "weight_dim_list: ", weight_dim_list
            print "theta_vec: ", theta_vec
            print "lambda_vec: ", lambda_vec
            w_array = np.reshape(w_array, w_array.shape[1])
            res_denominator = np.zeros(w_array.shape[0])
            for i in range(theta_vec.shape[0]):
                # print "gaussian theta i: ", i
                res_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array)
                if i == 0:
                    res_matrix = np.reshape(res_denominator_inc, (-1, 1))
                else:
                    res_matrix = np.concatenate((res_matrix, np.reshape(res_denominator_inc, (-1, 1))), axis=1)
                res_denominator = res_denominator + res_denominator_inc
            res_matrix = res_matrix / res_denominator.reshape((-1,1)).astype(float)
            print "np.sum(res_matrix, axis=1): ", np.sum(res_matrix, axis=1)
            ##### calculate responsibility #####

            w_update, theta_r_vec_update, lambda_t_vec_update = gaussian_mixture_gd_descent_avg(res_matrix, w_array, theta_r_vec, theta_vec, lambda_t_vec, lambda_vec, a, b, theta_alpha, C)
            weight_dim_list_index = 0
            weight_index = 0
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                if 'weight' in str(s):
                    param_dim_0 = weight_dim_list[weight_dim_list_index][0]
                    param_dim_1 = weight_dim_list[weight_dim_list_index][1]
                    param_regularization_grad = tensor.from_numpy(w_array[weight_index:(weight_index+param_dim_0 * param_dim_1)].reshape(weight_dim_list[weight_dim_list_index]))
                    param_regularization_grad.to_device(dev)
                    g = g + param_regularization_grad
                    opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
                    weight_dim_list_index = weight_dim_list_index + 1
                    weight_index = weight_index + param_dim_0 * param_dim_1
                else:
                    opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b) # bias update
            
            # update progress bar
            t_theta_r_vec = tensor.from_numpy(theta_r_vec)
            t_lambda_t_vec = tensor.from_numpy(lambda_t_vec)
            t_theta_r_vec_update = tensor.from_numpy(theta_r_vec_update)
            t_lambda_t_vec_update = tensor.from_numpy(lambda_t_vec_update)

            t_theta_r_vec.to_device(dev)
            t_lambda_t_vec.to_device(dev)
            t_theta_r_vec_update.to_device(dev)
            t_lambda_t_vec_update.to_device(dev)
            
            opt.apply_with_lr(epoch, get_lr(epoch)/1000., t_theta_r_vec_update, t_theta_r_vec, 't_theta_r_vec', b)
            opt.apply_with_lr(epoch, get_lr(epoch)/1000., t_lambda_t_vec_update, t_lambda_t_vec, 't_lambda_t_vec', b)
            
            t_theta_r_vec.to_device(cpudev)
            t_lambda_t_vec.to_device(cpudev)
            theta_r_vec = tensor.to_numpy(t_theta_r_vec)
            lambda_t_vec = tensor.to_numpy(t_lambda_t_vec)
            
            #############################################
            ## update theta_vec, theta_r_vec, lambda_vec, lambda_t_vec simultaneously!!!!!!!!
            theta_r_exp_vec = np.exp(theta_r_vec)
            theta_vec = theta_r_exp_vec / np.sum(theta_r_exp_vec)
            # lambda_vec = np.exp(lambda_t_vec)
            lambda_vec = np.exp(lambda_t_vec)
            print "theta_vec: ", theta_vec
            print "lambda_vec: ", lambda_vec
            #############################################

            utils.update_progress(b * 1.0 / num_train_batch,
                                  'training loss = %f, accuracy = %f' % (l, a))
        info = '\ntraining loss = %f, training accuracy = %f, lr = %f' \
            % (loss / num_train_batch, acc / num_train_batch, get_lr(epoch))
        print info

        loss, acc = 0.0, 0.0
        for b in range(num_test_batch):
            x = test_x[b * batch_size: (b + 1) * batch_size]
            y = test_y[b * batch_size: (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a

        print 'test loss = %f, test accuracy = %f' \
            % (loss / num_test_batch, acc / num_test_batch)
    net.save('model', 20)  # save model params into checkpoint file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dcnn for cifar10')
    parser.add_argument('model', choices=['vgg', 'alexnet', 'resnet', 'caffe'],
            default='alexnet')
    parser.add_argument('data', default='cifar-10-batches-py')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.data), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    print 'Loading data ..................'
    train_x, train_y = load_train_data(args.data)
    test_x, test_y = load_test_data(args.data)
    if args.model == 'caffe':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = caffe_net.create_net(args.use_cpu)
        # for cifar10_full_train_test.prototxt
        train((train_x, train_y, test_x, test_y), net, 160, alexnet_lr, 0.004,
              use_cpu=args.use_cpu)
        # for cifar10_quick_train_test.prototxt
        #train((train_x, train_y, test_x, test_y), net, 18, caffe_lr, 0.004,
        #      use_cpu=args.use_cpu)
    elif args.model == 'alexnet':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = alexnet.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 2, alexnet_lr, 0.004,
              use_cpu=args.use_cpu)
    elif args.model == 'vgg':
        train_x, test_x = normalize_for_vgg(train_x, test_x)
        net = vgg.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 250, vgg_lr, 0.0005,
              use_cpu=args.use_cpu)
    else:
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = resnet.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 200, resnet_lr, 1e-4,
              use_cpu=args.use_cpu)
