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
# (1) smoothness and non-negative are set to zero

import numpy as np
import os
import gzip
import argparse
import cPickle
from singa import initializer
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor


from singa.proto import core_pb2

def load_train_data(data_file, label_file, train_num, correl_file):
    file = open(data_file)
    data = np.genfromtxt(file, delimiter=",")
    file.close()
    file = open(label_file)
    label = np.genfromtxt(file, delimiter=",")
    file.close()
    idx = np.random.permutation(data.shape[0])
    traindata = data.astype(np.float32)
    trainlabel = label.astype(np.int)
    validdata = data.astype(np.float32)
    validlabel = label.astype(np.int)
    print traindata.shape, validdata.shape, trainlabel.shape, validlabel.shape
    file = open(correl_file)
    correldata = np.genfromtxt(file, delimiter=",") #this file is for the feature correlation matrix
    file.close()
    print correldata.shape
    correlmatrix = correldata.reshape((traindata.shape[1], traindata.shape[1])).astype(np.float32)
    print correldata.shape
    return traindata, validdata, trainlabel, validlabel, correlmatrix



def train(data_file, label_file, correl_file, h_dim, train_num, use_gpu, num_epoch=10, batch_size=100):
    print 'Start intialization............'
    lr = 0.1   # Learning rate
    weight_decay  = 0.0002
    hdim = h_dim
    print "hdim = \n", hdim
    nonnegative_constant = 0 # nonnegative penalty
    smooth_constant = 0 # smooth penalty


    #CHECK learning rates and momentum are not changed here
    opt = optimizer.SGD(momentum=0.5, weight_decay=weight_decay)

    print 'Loading data ..................'
    train_x, valid_x, train_y, valid_y, correlmatrix = load_train_data(data_file, label_file, train_num, correl_file)
    
    tcorrelmatrix = tensor.from_numpy(correlmatrix)

    num_train_batch = train_x.shape[0] / batch_size
    print "num_train_batch =  ", (num_train_batch)
    vdim = train_x.shape[1]
    print "vdim = \n", vdim
    
    #CHECK the following initilizations have been moved here after loading data
    tweight = tensor.from_numpy(np.zeros((vdim, hdim), dtype = np.float32))
    tvbias = tensor.from_numpy(np.zeros(vdim, dtype = np.float32))
    thbias = tensor.from_numpy(np.zeros(hdim, dtype = np.float32))

    #CHECK the following initializations have been moved here after tensor declarations
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

            #CHECK sparsity of the hidden

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
            error = tensor.sum(tensor.square((tdata - tnegvissample)))
            trainerrorsum = error + trainerrorsum
            # print 'error this batch = ', error
            sample_num = sample_num + tdata.shape[0]

            tgweight = tensor.mult(tnegvissample.T(), tneghidprob) -\
                    tensor.mult(tdata.T(), tposhidprob)
            tgvbias = tensor.sum(tnegvissample, 0) - tensor.sum(tdata, 0)
            print 'tgvbias sum(tdata): ', np.linalg.norm(tensor.to_numpy(tensor.sum(tdata, 0)))
            print 'tgvbias sum(tnegvissample): ', np.linalg.norm(tensor.to_numpy(tensor.sum(tnegvissample, 0)))
            
            tghbias = tensor.sum(tneghidprob, 0) - tensor.sum(tposhidprob, 0)
            print 'tgvbias sum(tposhidprob): ', np.linalg.norm(tensor.to_numpy(tensor.sum(tposhidprob, 0)))
            print 'tgvbias sum(tneghidprob): ', np.linalg.norm(tensor.to_numpy(tensor.sum(tneghidprob, 0)))
           
 
            print 'tgweight pos: ', np.linalg.norm(tensor.to_numpy(tensor.mult(tdata.T(), tposhidprob)))
            print 'tgweight neg: ', np.linalg.norm(tensor.to_numpy(tensor.mult(tnegvissample.T(), tneghidprob)))
            print 'error this batch = ', error

            print 'lr: ', lr
            print 'weight_decay: ', weight_decay
            # print 'momentum: ', momentum
            print '\n'
            tweightzero = tensor.from_numpy(np.zeros((vdim, hdim), dtype = np.float32))
            tweightltzero = tensor.gt(tweight, tweightzero)
            #CHECK order of substraction
            #CHECK multiplication * 
            #CHECK no lr here
            tgweight = tgweight + nonnegative_constant * tweightltzero 

            #CHECK no lr here even in matlab code
            tgweight = tgweight + smooth_constant * tensor.mult(tcorrelmatrix, tweight)

            # print 'tgweight norm: ', np.linalg.norm(tensor.to_numpy(tgweight))
            # print 'tgvbias norm: ', np.linalg.norm(tensor.to_numpy(tgvbias))
            # print 'tghbias norm: ', np.linalg.norm(tensor.to_numpy(tghbias))

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
    parser.add_argument('datafile', type=str, help='the dataset path')
    parser.add_argument('labelfile', type=str, help='the datalabel path')
    parser.add_argument('correlfile', type=str, help='the correlation matrix path')
    parser.add_argument('hdim', type=int, help='hidden dimension')
    parser.add_argument('trainnum', type=int, help='the number of training samples')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.datafile), 'Pls check the data file'
    assert os.path.exists(args.labelfile), 'Pls check the label file'
    assert os.path.exists(args.correlfile), 'Pls check the correl file'
    validposhidprob, valid_y = train(args.datafile, args.labelfile, args.correlfile, args.hdim, args.trainnum, args.use_gpu)
    print "validposhidprob shape = \n", validposhidprob.shape
    print "valid_y shape = \n", valid_y.shape
    # save the transformed features and label files for next step logistic regression
    a = np.asarray(validposhidprob, dtype = float)
    b= np.asarray(valid_y, dtype = int)
    np.savetxt('transformed_feature.csv', a, fmt = '%6f', delimiter=",")
    np.savetxt('label.csv', b, fmt = '%d', delimiter=",")
#running script: python enRBM-SINGA.py /data/zhaojing/regularization/CMSHF/CMS_HF_VECTOR_Regulariz_diag_features.txt /data/zhaojing/regularization/CMSHF/CMS_HF_VECTOR_Regulariz_label.csv /data/zhaojing/regularization/CMSHF/CMSHFSimilarityMatrix2level.txt 100 600
