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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================

# all the data are read from csv not from URL
# occlude test are added here

import sys, os
import traceback
import time
import urllib
import numpy as np
from argparse import ArgumentParser


from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType

import model


def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train Alexnet over CIFAR10")

        parser.add_argument('-p', '--port', default=9989, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=140)

        # Process arguments
        args = parser.parse_args()
        port = args.port

        use_cpu = args.use_cpu
        if use_cpu:
            print "runing with cpu"
            dev = device.get_default_device()
        else:
            print "runing with gpu"
            dev = device.create_cuda_gpu()

        # start to train
        agent = Agent(port)
        train(dev, agent, args.max_epoch, use_cpu)
        # wait the agent finish handling http request
        agent.stop()
    except SystemExit:
        return
    except:
        # p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")


def get_train_data(train_sample_url, train_label_url):
    '''load data'''
    train_sample = np.genfromtxt(train_sample_url, delimiter=',')
    train_label = np.genfromtxt(train_label_url, delimiter=',')
    return train_sample, train_label

def get_test_data(test_sample_url, test_label_url):
    '''load data'''
    test_sample = np.genfromtxt(test_sample_url, delimiter=',')
    test_label = np.genfromtxt(test_label_url, delimiter=',')
    return test_sample, test_label

def get_occlude_data(test_sample_url, test_label_url, width_idx, kernel_y, kernel_x, stride_y, stride_x):
    '''load occlude data'''
    occlude_data = np.genfromtxt(file_url, delimiter=',')
    for j in range(kernel_y):
        for i in range ((height_idx * stride_y + j) * width + width_idx * stride_x, ((height_idx * stride_y + j) * width + width_idx * stride_x + kernel_x)):
            occlude_data[i] = float(0.0)
    return occlude_data

def handle_cmd(agent):
    pause = False
    stop = False
    while not stop:
        key, val = agent.pull()
        if key is not None:
            msg_type = MsgType.parse(key)
            if msg_type.is_command():
                if MsgType.kCommandPause.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    pause = True
                elif MsgType.kCommandResume.equal(msg_type):
                    agent.push(MsgType.kStatus, "Success")
                    pause = False
                elif MsgType.kCommandStop.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    stop = True
                else:
                    agent.push(MsgType.kStatus,"Warning, unkown message type")
                    print "Unsupported command %s" % str(msg_type)
        if pause and not stop:
            time.sleep(0.1)
        else:
            break
    return stop


def get_lr(epoch):
    '''change learning rate as epoch goes up'''
    return 0.001


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

def train(dev, agent, max_epoch, use_cpu, batch_size=100):

    opt = optimizer.SGD(momentum=0.8, weight_decay=0.01)

    agent.push(MsgType.kStatus, 'Downlaoding data...')
    train_sample, train_label = get_train_data\
    ('/data/zhaojing/regularization/LACE-CNN-1500/reverse-order/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_12slots_reverse.csv', \
    '/data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv')  # PUT THE DATA on/to dbsystem
    agent.push(MsgType.kStatus, 'Finish downloading data')
    test_sample, test_label = get_test_data\
    ('/data/zhaojing/regularization/LACE-CNN-1500/reverse-order/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_12slots_reverse.csv', \
    '/data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv')  # PUT THE DATA on/to dbsystem
    tx = tensor.Tensor((batch_size, in_shape[0], in_shape[1], in_shape[2]), dev)
    ty = tensor.Tensor((batch_size, ), dev, core_pb2.kInt)
    num_train_batch = train_sample.shape(0) / batch_size
#    num_test_batch = test_x.shape[0] / (batch_size)
    idx = np.arange(len(records), dtype=np.int32)

    net = model.create_net(in_shape, use_cpu)
    net.to_device(dev)
    hiehgt = 12
    width = 375
    kernel_y = 3
    kernel_x = 80
    stride_y = 1
    stride_x = 20

    for epoch in range(max_epoch):
        if handle_cmd(agent):
            break
        np.random.seed(10)
        np.random.shuffle(idx)
        print 'Epoch %d' % epoch
        
        if epoch % test_epoch == 10:
            loss, acc = 0.0, 0.0
            x, y = test_sample[b * batch_size:(b + 1) * batch_size], test_label[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a, probs = net.evaluate(tx, ty)
            loss += l
            acc += a
            print 'testing loss = %f, accuracy = %f' % (loss / num_test_batch,
                                                        acc / num_test_batch)
            # put test status info into a shared queue
            info = dict(
                phase='test',
                step = epoch,
                accuracy = acc / num_test_batch,
                loss = loss / num_test_batch,
                timestamp = time.time())
            agent.push(MsgType.kInfoMetric, info)

        if epoch % occlude_test_epoch == 100:
            # occlude test data
            height_dim = (height - kernel_y) / stride_y + 1; # 10
            width_dim = (width - kernel_x) / stride_x + 1; # 60
            true_label_prob_matrix = np.zeros([(height_dim * width_dim), 1])
            for height_idx in range(height_dim):
                for width_idx in range(width_dim):
                    occlude_test_sample, occlude_test_label = get_occlude_data('/data/zhaojing/regularization/LACE-CNN-1500/reverse-order/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_12slots_reverse.csv', \
                    '/data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv', \
                    height_idx, width_idx, kernel_y, kernel_x, stride_y, stride_x)
                    loss, acc = 0.0, 0.0
                    x, y = occlude_test_records # !!! where are the labels?
                    tx.copy_from_numpy(x)
                    ty.copy_from_numpy(y)
                    l, a, probs = net.evaluate(tx, ty)
                    y_scores = softmax(tensor.to_numpy(probs))[:,1]
                    for i in range(0, x.shape(0)): # !!! y_scores ~~ first is o then is 1 !!!
                        if y_true[i] == 1:
                            sum_true_label_prob = sum_true_label_prob + y_scores[i, 1]
                        elif y_true[i] == 0:
                            sum_true_label_prob = sum_true_label_prob + (1 - y_scores[i, 1])
                    true_label_prob_matrix[height_idx * width_dim + width_idx, 0] = sum_true_label_prob / x.shape(0)


        loss, acc = 0.0, 0.0
        for b in range(num_train_batch):
            x, y = train_sample[b * batch_size:(b + 1) * batch_size], train_label[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a), probs = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_specs(),
                                 net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s.name))
            info = 'training loss = %f, training accuracy = %f' % (l, a)
            utils.update_progress(b * 1.0 / num_train_batch, info)
        # put training status info into a shared queue
        info = dict(phase='train', step=epoch,
                    accuracy=acc/num_train_batch,
                    loss=loss/num_train_batch,
                    timestamp=time.time())
        agent.push(MsgType.kInfoMetric, info)
        info = 'training loss = %f, training accuracy = %f' \
            % (loss / num_train_batch, acc / num_train_batch)
        print info
        print "probs shape: ", tensor.to_numpy(probs).shape
        print "probs for readmitted: ", softmax(tensor.to_numpy(probs))[:,1]

        if epoch > 0 and epoch % 30 == 0:
            net.save('parameter_%d' % epoch)
    net.save('parameter_last')


if __name__ == '__main__':
    main()