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
# data normalziation (zero mean, unit variance)

import cPickle
import numpy as np
import os
import argparse
import datetime

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2


import readmissionnet
import conf

# readmission net
def rnet_lr(epoch):
    return 0.05 / float(1 << (epoch / 25))


def load_data(data_dir_path, label_dir_path):
    data = np.genfromtxt(data_dir_path, delimiter=',')
    labels = np.genfromtxt(label_dir_path, delimiter=',')
    return data, labels


def train(lr, ssfolder, data, net, max_epoch, get_lr, weight_decay, input_shape, test_batch_size, batch_size=100,
          use_cpu=False):
    print 'Start intialization............'
    if use_cpu:
        print 'Using CPU'
        dev = device.get_default_device()
    else:
        print 'Using GPU'
        dev = device.create_cuda_gpu()

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    ttrainx = tensor.Tensor((batch_size,) + input_shape, dev)
    ttrainy = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
    ttestx = tensor.Tensor((test_batch_size,) + input_shape, dev)
    ttesty = tensor.Tensor((test_batch_size,), dev, core_pb2.kInt)
    train_x, train_y, test_x, test_y = data
    num_train_batch = train_x.shape[0] / batch_size
    num_test_batch = test_x.shape[0] / test_batch_size
    # remainder = num_test % batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    best_auc = 0.0
    best_loss = 0.0
    nb_epoch_for_best_auc = 0

    for epoch in range(max_epoch):
        np.random.shuffle(idx)
        loss, auc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
            ttrainx.copy_from_numpy(x)
            ttrainy.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
            # update progress bar
            utils.update_progress(b * 1.0 / num_train_batch,
                                  'training loss = %f' % l)
        info = '\ntraining loss = %f, lr = %f' \
            % (loss / num_train_batch, get_lr(epoch))
        print info

        loss, auc = 0.0, 0.0
        for b in range(num_test_batch):
            x = test_x[b * test_batch_size: (b + 1) * test_batch_size]
            y = test_y[b * test_batch_size: (b + 1) * test_batch_size]
            print "test x shape: ", x.shape
            print "test y shape: ", y.shape
            ttestx.copy_from_numpy(x)
            ttesty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            auc = a

        print 'test loss = %f, test auc = %f' \
            % (loss / num_test_batch, auc)

        if auc > best_auc + 0.005:
            best_auc = auc
            best_loss = loss
            nb_epoch_for_best_auc = 0
        else:
            nb_epoch_for_best_auc += 1
            if nb_epoch_for_best_auc > 8:
                break
            elif nb_epoch_for_best_auc % 4 ==0:
                lr /= 10
                logging.info("Decay the learning rate from %f to %f" %(lr*10, lr))

    #net.save('model', 20)  # save model params into checkpoint file
    net.save(str(os.path.join(ssfolder, 'model')), buffer_size=200)
    return (best_auc, best_loss)

if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='Train dcnn for cifar10')
#    parser.add_argument('model', choices=['vgg', 'alexnet', 'resnet', 'caffe'],
#            default='alexnet')
#    parser.add_argument('data', default='cifar-10-batches-py')
    parser.add_argument('-traindata', type=str, help='the traindata path')
    parser.add_argument('-testdata', type=str, help='the testdata path')
    parser.add_argument('-testbatchsize', type=int, help='testbatchsize')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()
    
    cnf = conf.Conf()
    log_dir = os.path.join(cnf.log_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)

    best_auc = 0.0
    best_loss = 0
    best_idx = -1

    train_x, train_y = load_data(args.traindata)
    test_x, test_y = load_data(args.testdata)

    #elif args.model == 'alexnet':
    #    train_x, test_x = normalize_for_alexnet(train_x, test_x)
    #    net = alexnet.create_net(args.use_cpu)
    #    train((train_x, train_y, test_x, test_y), net, 2, alexnet_lr, 0.004,
    #          use_cpu=args.use_cpu)
    for i in range(30):
        ssfolder = cnf.snapshot_folder + str(i)
        if not os.path.isdir(ssfolder):
            os.makedirs(ssfolder)
        cnf.gen_conf()
        with open(os.path.join(log_dir, '%d.conf' % i), 'w') as fconf:
            cnf.dump(fconf)
        
        net = readmissionnet.create_net(cnf.input_shape, cnf.use_cpu)
        logging.info('The %d-th trial' % i)
        auc,loss= train(cnf.lr, ssfolder, (train_x, train_y, test_x, test_y), cnf.input_folder, net,
                      cnf.num_epoch, rnet_lr, cnf.decay, args.testbatchsize, cnf.input_shape, cnf.batch_size, cnf.use_cpu)
        logging.info('The best test auc for %d-th trial is %f, with loss=%f' % (i, auc, loss))
        if best_auc < auc:
            best_auc = auc
            best_loss = loss
            best_idx = i
        logging.info('The best test auc so far is %f, with loss=%f, for the %d-th conf'
                    % (best_auc, best_loss, best_idx))
