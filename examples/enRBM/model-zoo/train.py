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

import numpy as np
from numpy import linalg as LA
import os
import gzip
import argparse
import cPickle
from singa import initializer
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
import data


from singa.proto import core_pb2

def matrix_1_norm(matrix):
    normsum = 0.0
    for i in range(matrix.shape[0]):
        normsum = normsum + LA.norm(matrix[i, :], 1)
    return normsum

def train(h_dim, train_num, use_gpu, num_epoch=10, batch_size=100):
    print 'Start intialization............'
    lr = 0.1   # Learning rate
    weight_decay  = 0.0002
    hdim = h_dim
    print "hdim = \n", hdim
    nonnegative_constant = 0.0 # nonnegative penalty
    smooth_constant = 0.0 # smooth penalty


    #CHECK learning rates and momentum are not changed here
    opt = optimizer.SGD(momentum=0.5, weight_decay=weight_decay)

    print 'Loading data ..................'
    train_x, valid_x, train_y, valid_y, correlmatrix = data.load_dataset(train_num)
    tcorrelmatrix = tensor.from_numpy(correlmatrix)
    num_train_batch = train_x.shape[0] / batch_size
    print "num_train_batch =  ", (num_train_batch)
    vdim = train_x.shape[1]
    print "vdim = \n", vdim
    
    tweight = tensor.from_numpy(np.zeros((vdim, hdim), dtype = np.float32))
    tvbias = tensor.from_numpy(np.zeros(vdim, dtype = np.float32))
    thbias = tensor.from_numpy(np.zeros(hdim, dtype = np.float32))

    if use_gpu:
        dev = device.create_cuda_gpu()
    else:
        dev = device.get_default_device()

    for t in [tweight, tvbias, thbias, tcorrelmatrix]:
        t.to_device(dev)

    for epoch in range(num_epoch):
        trainerrorsum = 0.0
        sample_num = 0
        print 'Epoch ', epoch
        for b in range(num_train_batch):
            # positive phase
            tdata = tensor.from_numpy(
                    train_x[(b * batch_size):((b + 1) * batch_size), : ])
            tdata.to_device(dev)
            tposhidprob = tensor.mult(tdata, tweight)
            tposhidprob.add_row(thbias)
            tposhidprob = tensor.sigmoid(tposhidprob)

            # CHECK sparsity of the hidden
            tposhidrandom = tensor.Tensor(tposhidprob.shape, dev)
            tposhidrandom.uniform(0.0, 1.0)
            tposhidsample = tensor.gt(tposhidprob, tposhidrandom)

            # negative phase
            tnegvisprob = tensor.mult(tposhidsample, tweight.T())
            tnegvisprob.add_row(tvbias)
            tnegvisprob = tensor.sigmoid(tnegvisprob)

            # sample in vis layer
            tnegvisrandom = tensor.Tensor(tnegvisprob.shape, dev)
            tnegvisrandom.uniform(0.0, 1.0)
            tnegvissample = tensor.gt(tnegvisprob, tnegvisrandom)

            tneghidprob = tensor.mult(tnegvissample, tweight)
            tneghidprob.add_row(thbias)
            tneghidprob = tensor.sigmoid(tneghidprob)
            error = tensor.sum(tensor.abs((tdata - tnegvissample)))
            trainerrorsum = error + trainerrorsum
            sample_num = sample_num + tdata.shape[0]

            tgweight = tensor.mult(tnegvissample.T(), tneghidprob) -\
                    tensor.mult(tdata.T(), tposhidprob)
            tgvbias = tensor.sum(tnegvissample, 0) - tensor.sum(tdata, 0)
            tghbias = tensor.sum(tneghidprob, 0) - tensor.sum(tposhidprob, 0)
 
            # non-negative check 
            tweightzero = tensor.from_numpy(np.zeros((vdim, hdim), dtype = np.float32))
            tweightltzero = tensor.gt(tweight, tweightzero)
            tgweight = tgweight + nonnegative_constant * tweightltzero 
            tgweight = tgweight + smooth_constant * tensor.mult(tcorrelmatrix, tweight)

            opt.apply_with_lr(epoch, lr / batch_size, tgweight, tweight, 'w')
            opt.apply_with_lr(epoch, lr / batch_size, tgvbias, tvbias, 'vb')
            opt.apply_with_lr(epoch, lr / batch_size, tghbias, thbias, 'hb')

        print 'training errorsum = ', (trainerrorsum)
        print 'training sample number = ', (sample_num)

        tvaliddata = tensor.from_numpy(valid_x)
        tvaliddata.to_device(dev)
        tvalidposhidprob = tensor.mult(tvaliddata, tweight)
        tvalidposhidprob.add_row(thbias)
        tvalidposhidprob = tensor.sigmoid(tvalidposhidprob)
        tvalidposhidrandom = tensor.Tensor(tvalidposhidprob.shape, dev)
        initializer.uniform(tvalidposhidrandom, 0.0, 1.0)
        tvalidposhidsample = tensor.gt(tvalidposhidprob, tvalidposhidrandom)
        tvalidnegvisprob = tensor.mult(tvalidposhidsample, tweight.T())
        tvalidnegvisprob.add_row(tvbias)
        tvalidnegvisprob = tensor.sigmoid(tvalidnegvisprob)

        # sample in vis layer
        tvalidnegvisrandom = tensor.Tensor(tvalidnegvisprob.shape, dev)
        tvalidnegvisrandom.uniform(0.0, 1.0)
        tvalidnegvissample = tensor.gt(tvalidnegvisprob, tvalidnegvisrandom)
        validerrorsum = tensor.sum(tensor.square((tvaliddata - tvalidnegvissample)))
        print 'valid errorsum = ', (validerrorsum)
        
    return tensor.to_numpy(tvalidposhidprob), valid_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RBM over MNIST')
    parser.add_argument('hdim', type=int, help='hidden dimension')
    parser.add_argument('trainnum', type=int, help='the number of training samples')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    validposhidprob, valid_y = train(args.hdim, args.trainnum, args.use_gpu)
    print "validposhidprob shape = \n", validposhidprob.shape
    print "valid_y shape = \n", valid_y.shape
    a = np.asarray(validposhidprob, dtype = float)
    b= np.asarray(valid_y, dtype = int)
    np.savetxt('transformed_feature.csv', a, fmt = '%6f', delimiter=",")
    np.savetxt('label.csv', b, fmt = '%d', delimiter=",")
#running script: python enRBM-SINGA-debug.py 200 600
