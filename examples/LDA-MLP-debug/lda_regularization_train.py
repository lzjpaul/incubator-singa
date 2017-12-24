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

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2

import lda_regularization_optimizer
import lda_mlp_model

import datetime
import time
import random

def load_all_data(featurefilepath, labelfilepath):
    all_x = np.genfromtxt(featurefilepath, dtype=np.float32, delimiter=',')
    all_y = np.genfromtxt(labelfilepath, dtype=np.float32, delimiter=',')
    print 'after loading all data'
    return all_x, all_y

def load_train_data(featurefilepath, labelfilepath):
    train_x = np.genfromtxt(featurefilepath, dtype=np.float32, delimiter=',')
    train_y = np.genfromtxt(labelfilepath, dtype=np.float32, delimiter=',')
    print 'after loading train data'
    return train_x, train_y

def load_test_data(featurefilepath, labelfilepath):
    test_x = np.genfromtxt(featurefilepath, dtype=np.float32, delimiter=',')
    test_y = np.genfromtxt(labelfilepath, dtype=np.float32, delimiter=',')
    print 'after lodading test data'
    return test_x, test_y

def normalize_for_vgg(train_x, test_x):
    mean = train_x.mean()
    std = train_x.std()
    train_x -= mean
    test_x -= mean
    train_x /= std
    test_x /= std
    return train_x, test_x

def mlp_lr(epoch):
    return 0.001

def train(data, model_name, hyperpara, ldapara, phi, uptfreq, net, max_epoch, get_lr, weight_decay, gpuid, batch_size=10,
          use_cpu=False):
    print 'Start intialization............'
    if use_cpu:
        print 'Using CPU'
        dev = device.get_default_device()
        cpudev = dev
    else:
        print 'Using GPU'
        dev = device.create_cuda_gpu_on(gpuid)
        cpudev = device.get_default_device()

    net.to_device(dev)
    opt = lda_regularization_optimizer.LDASGD(net=net, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        print 'param names: ', p
        opt.register(p, specs)
    for (s, p) in zip(net.param_names(), net.param_values()):
        if s == 'dense1/weight': #only one lda_regularizer is created
            opt.lda_register(hyperpara, ldapara, phi, uptfreq)

    train_x, train_y, test_x, test_y = data
    ttrainx = tensor.Tensor((batch_size, train_x.shape[1]), dev)
    # ttrainy = tensor.Tensor((batch_size, train_y.shape[1]), dev, core_pb2.kInt)
    ttrainy = tensor.Tensor((batch_size, train_y.shape[1]), dev)
    ttestx = tensor.Tensor((test_x.shape[0], test_x.shape[1]), dev)
    # ttesty = tensor.Tensor((test_x.shape[0], test_y.shape[1]), dev, core_pb2.kInt)
    ttesty = tensor.Tensor((test_x.shape[0], test_y.shape[1]), dev)
    num_train_batch = train_x.shape[0] / batch_size
    print 'num_train_batch: ', num_train_batch
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    test_epoch = 1
    for epoch in range(max_epoch):
        np.random.seed(epoch)
        np.random.shuffle(idx)
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
            ttrainx.copy_from_numpy(x)
            ttrainy.copy_from_numpy(y)
            grads, (l, a), probs = net.train(ttrainx, ttrainy)
            loss += l
            acc += a[0]
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                # print 'name: ', s
                # print 'p l2: ', p.l2()
                # print '0.001g l2', (0.001*g).l2()
                opt.apply_with_lr(dev=dev, trainnum=train_x.shape[0], net=net, epoch=epoch, lr=get_lr(epoch), grad=g, value=p, name=str(s), step=b)
        info = '\ntraining loss = %f, training accuracy = %f, lr = %f' \
            % (loss / num_train_batch, acc / num_train_batch, get_lr(epoch))
        print info

        if epoch % test_epoch == 0 or epoch == (max_epoch-1):
            loss, acc, macro_auc, micro_auc = 0.0, 0.0, 0.0, 0.0
            x, y = np.copy(test_x), np.copy(test_y)
            ttestx.copy_from_numpy(x)
            ttesty.copy_from_numpy(y)
            l, a, probs = net.evaluate(ttestx, ttesty)
            loss += l
            acc += a[0]
            macro_auc += a[1]
            micro_auc += a[2]
            print 'test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f' \
                % (loss, acc, macro_auc, micro_auc)
            if epoch == (max_epoch - 1):
                print 'final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f' \
                % (loss, acc, macro_auc, micro_auc)
                # write_out_result(resultpath, hyperpara_list, hyperpara_idx, gm_num, gm_lambda_ratio, uptfreq, get_lr(epoch), weight_decay, batch_size, loss / num_test_batch, acc / num_test_batch)      
    model_time = time.time()
    model_time = datetime.datetime.fromtimestamp(model_time).strftime('%Y-%m-%d-%H-%M-%S')
    print 'model time: ', model_time
    net.save('model-time-' + model_time, 20)  # save model params into checkpoint file
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mlp for healthcare dataset')
    parser.add_argument('model', choices=['ldamlp'],
            default='alexnet')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    parser.add_argument('-topicnum', type=int, help='topic_number')
    parser.add_argument('-ldauptfreq', type=int, help='lda update frequency, in steps')
    parser.add_argument('-paramuptfreq', type=int, help='parameter update frequency, in steps')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    #parser.add_argument('-resultpath', type=str, help='result path')
    args = parser.parse_args()
    print 'Loading data ..................'
    
    all_x, all_y = load_all_data('data-repository/feature_matrix_try.csv', 'data-repository/result_matrix_try.csv')
    train_num = int(all_x.shape[0] * 0.8)
    train_x, train_y = all_x[0:train_num], all_y[0:train_num]
    test_x, test_y = all_x[train_num:all_x.shape[0]], all_y[train_num:all_x.shape[0]]
    
    ''' 
    train_x, train_y = load_train_data('data-repository/train_x.csv', 'data-repository/train_y.csv')
    test_x, test_y = load_test_data('data-repository/test_x.csv', 'data-repository/test_y.csv')
    '''
    print 'train number: ', train_x.shape[0]
    print 'test number: ', test_x.shape[0]
    alpha = 1 + 0.05
    phi = np.genfromtxt('data-repository/phi.csv', delimiter=',')
    phi = np.transpose(phi)
    if args.model == 'ldamlp':
        start = time.time()
        st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print st
        max_epoch = args.maxepoch
        doc_num = 128 # hard-code, number of hidden units 
        topic_num = args.topicnum
        word_num = train_x.shape[1]
        net = lda_mlp_model.create_net((word_num,), args.use_cpu)
        train((train_x, train_y, test_x, test_y), args.model, [alpha], [doc_num, topic_num, word_num], phi, [args.ldauptfreq, args.paramuptfreq], net, args.maxepoch, mlp_lr, 0.0, args.gpuid, use_cpu=args.use_cpu)
        done = time.time()
        do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
        print do
        elapsed = done - start
        print elapsed
