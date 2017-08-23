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

""" Current problems:
(1) calresponsibility for high-dimensional non-sparse vector is time-consuming
(3) (done!!) original weight may recieve L2 norm regularization, then it will receive both L2 norm and GM regularization ???
(4) (done!!) would it go into apply_with_lr becuase I override apply_with_lr with different signatures ???
(5) check logic because I deleted so many "prints"
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
from caffe import caffe_net

import gm_prior_optimizer
import alexnet
import vgg
import resnet
import gm_prior_data as dt

import datetime
import time
import random


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


def normalize_for_resnet(train_x, test_x):
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


def train(data, hyperpara_list, hyperpara_idx, gm_num, gm_lambda_ratio, uptfreq, net, max_epoch, get_lr, weight_decay, gpuid, batch_size=100,
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
    opt = gm_prior_optimizer.GMSGD(net=net, momentum=0.9, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)
    for (s, p) in zip(net.param_names(), net.param_values()):
        opt.gm_register(s, p, hyperpara_list, hyperpara_idx, gm_num, gm_lambda_ratio, uptfreq)
    opt.weightdimSum = sum(opt.weight_dim_list.values())
    print "opt.weightdimSum: ", opt.weightdimSum
    print "opt.weight_name_list: ", opt.weight_name_list
    print "opt.weight_dim_list: ", opt.weight_dim_list


    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
    train_x, train_y, test_x, test_y = data
    dl_train = dt.CifarBatchIter(train_x, train_y, batch_size, dt.numpy_crop_flip,
            shuffle=True, capacity=10)
    dl_train.start()
    num_train = dl_train.num_samples
    num_train_batch = num_train / batch_size
    num_test_batch = test_x.shape[0] / batch_size
    print 'num_train, num_train_batch, num_test_batch: ', num_train, num_train_batch, num_test_batch
    for epoch in range(max_epoch):
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            x, y = dl_train.next()
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                opt.apply_with_lr(dev=dev, trainnum=train_x.shape[0], net=net, epoch=epoch, lr=get_lr(epoch), grad=g, value=p, name=str(s), step=b)
            # update progress bar
            # utils.update_progress(b * 1.0 / num_train_batch,
            #                       'training loss = %f, accuracy = %f' % (l, a))
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
        if epoch == (max_epoch - 1):
            print 'final test loss = %f, test accuracy = %f' \
            % (loss / num_test_batch, acc / num_test_batch)
    dl_train.end()
    net.save('model', 20)  # save model params into checkpoint file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dcnn for cifar10')
    parser.add_argument('model', choices=['vgg', 'alexnet', 'resnet', 'caffe'],
            default='alexnet')
    parser.add_argument('data', default='cifar-10-batches-py')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    parser.add_argument('-gmnum', type=int, help='gm_number')
    parser.add_argument('-gmuptfreq', type=int, help='gm update frequency, in steps')
    parser.add_argument('-paramuptfreq', type=int, help='parameter update frequency, in steps')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    args = parser.parse_args()
    assert os.path.exists(args.data), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    print 'Loading data ..................'
    train_x, train_y = load_train_data(args.data)
    test_x, test_y = load_test_data(args.data)
    # decay_array = np.array([0.01, 0.001, 0.0001]) #other parameters like bias may need weight_decay in the implementations
    # momentum_array = np.array([0.8, 0.9])
    b_list, alpha_list = [0.03, 0.01, 0.001, 0.0001],\
                   [0.3]
    a_list = [1e-1]
    b_val_num = len(b_list)
    alpha_val_num = len(alpha_list)
    a_val_num = len(a_list)
    gm_lambda_ratio_list = [ -1.]
    # gm_lambda_ratio_list = [-1.]
    gm_lambda_ratio = random.choice(gm_lambda_ratio_list)
    if args.model == 'caffe':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        b_idx_arr, alpha_idx_arr, a_idx_arr = np.arange(b_val_num), np.arange(alpha_val_num), np.arange(a_val_num)
        for alpha_idx in alpha_idx_arr:
            for b_idx in b_idx_arr:
                for a_idx in a_idx_arr:
                    start = time.time()
                    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                    print st
                    print "train_x shape: ", train_x.shape
                    print "train_x norm: ", np.linalg.norm(train_x)
                    max_epoch = args.maxepoch
                    gm_num = args.gmnum
                    net = caffe_net.create_net(args.use_cpu)
                    # for cifar10_full_train_test.prototxt
                    train((train_x, train_y, test_x, test_y), [a_list, b_list, alpha_list], [a_idx, b_idx, alpha_idx], gm_num, gm_lambda_ratio, [args.gmuptfreq, args.paramuptfreq], net, 160, alexnet_lr, 0.004, args.gpuid, use_cpu=args.use_cpu)
                    # for cifar10_quick_train_test.prototxt
                    #train((train_x, train_y, test_x, test_y), net, 18, caffe_lr, 0.004,
                    #      use_cpu=args.use_cpu)
                    done = time.time()
                    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                    print do
                    elapsed = done - start
                    print elapsed
    elif args.model == 'alexnet':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        b_idx_arr, alpha_idx_arr, a_idx_arr = np.arange(b_val_num), np.arange(alpha_val_num), np.arange(a_val_num)
        for alpha_idx in alpha_idx_arr:
            for b_idx in b_idx_arr:
                for a_idx in a_idx_arr:
                    start = time.time()
                    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                    print st
                    print "train_x shape: ", train_x.shape
                    print "train_x norm: ", np.linalg.norm(train_x)
                    max_epoch = args.maxepoch
                    gm_num = args.gmnum
                    net = alexnet.create_net(args.use_cpu)
                    train((train_x, train_y, test_x, test_y), [a_list, b_list, alpha_list], [a_idx, b_idx, alpha_idx], gm_num, gm_lambda_ratio, [args.gmuptfreq, args.paramuptfreq], 
                          net, 160, alexnet_lr, 0.004, args.gpuid, use_cpu=args.use_cpu)
                    done = time.time()
                    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                    print do
                    elapsed = done - start
                    print elapsed
    elif args.model == 'vgg':
        train_x, test_x = normalize_for_vgg(train_x, test_x)
        b_idx_arr, alpha_idx_arr, a_idx_arr = np.arange(b_val_num), np.arange(alpha_val_num), np.arange(a_val_num)
        for alpha_idx in alpha_idx_arr:
            for b_idx in b_idx_arr:
                for a_idx in a_idx_arr:
                    start = time.time()
                    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                    print st
                    max_epoch = args.maxepoch
                    gm_num = args.gmnum
                    net = vgg.create_net(args.use_cpu)
                    train((train_x, train_y, test_x, test_y), [a_list, b_list, alpha_list], [a_idx, b_idx, alpha_idx], gm_num, gm_lambda_ratio, [args.gmuptfreq, args.paramuptfreq], 
                          net, 250, vgg_lr, 0.0005, args.gpuid, use_cpu=args.use_cpu)
                    done = time.time()
                    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                    print do
                    elapsed = done - start
                    print elapsed
    else:
        train_x, test_x = normalize_for_resnet(train_x, test_x)
        b_idx_arr, alpha_idx_arr, a_idx_arr = np.arange(b_val_num), np.arange(alpha_val_num), np.arange(a_val_num)
        for alpha_idx in alpha_idx_arr:
            for b_idx in b_idx_arr:
                for a_idx in a_idx_arr:
                    start = time.time()
                    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
                    print st
                    max_epoch = args.maxepoch
                    gm_num = args.gmnum
                    net = resnet.create_net(args.use_cpu)
                    train((train_x, train_y, test_x, test_y), [a_list, b_list, alpha_list], [a_idx, b_idx, alpha_idx], gm_num, gm_lambda_ratio, [args.gmuptfreq, args.paramuptfreq], 
                          net, 200, resnet_lr, 1e-4, args.gpuid, use_cpu=args.use_cpu)
                    done = time.time()
                    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
                    print do
                    elapsed = done - start
                    print elapsed
