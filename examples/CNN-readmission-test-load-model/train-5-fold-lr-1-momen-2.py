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

# modify
# 6-20: occlude: consider samples as well
import sys, os
import traceback
import time
import urllib
import numpy as np
from argparse import ArgumentParser
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import random
from random import randint

from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType
from data_loader import *
from explain_occlude_area import *
from explain_occlude_area_format_out import *
import model


def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train CNN Readmission Model")

        parser.add_argument('-p', '--port', default=9989, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=600)

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

def get_occlude_data(occlude_feature, occlude_label, height, width, height_idx, width_idx, kernel_y, kernel_x, stride_y, stride_x):
    '''load occlude data'''
    for n in range(occlude_feature.shape[0]): #sample
        for j in range (kernel_y):
            occlude_feature[n, ((height_idx * stride_y + j) * width + width_idx * stride_x) : ((height_idx * stride_y + j) * width + width_idx * stride_x + kernel_x)] = float(0.0)
    return occlude_feature, occlude_label

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
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

def cal_accuracy(yPredict, yTrue):
    return np.sum(((yPredict > 0.5) == yTrue).astype(int)) / float(yTrue.shape[0])

def auroc(yPredictProba, yTrue):
    return roc_auc_score(yTrue, yPredictProba)

def train(dev, agent, max_epoch, use_cpu, batch_size=100):
    lr_array = np.array([0.001])
    decay_array = np.array([0.01, 0.001, 0.0001])
    momentum_array = np.array([0.9])
    kernel_y_param_array = np.array([2, 3])
    kernel_x_param_array = np.array([6, 10, 15, 20, 25, 30, 35, 40, 65, 80, 100])
    stride_y_param_array = 1
    stride_x_param_array = np.array([3, 5, 8, 10])
    all_feature, all_label = get_data\
    ('/data/zhaojing/regularization/LACE-CNN-1500/reverse-order/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_12slots_reverse.csv', \
    '/data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv')  # PUT THE DATA on/to dbsystem
    
    for i in range(len(lr_array)):
        lr_param = lr_array[i]
        for j in range(len(decay_array)):
            decay_param = decay_array[j]
            for k in range(len(momentum_array)):
                momentum_param = momentum_array[k]
                kernel_x_param = kernel_x_param_array[random.randint(0,len(kernel_x_param_array)-1)]
                kernel_y_param = kernel_y_param_array[random.randint(0,len(kernel_y_param_array)-1)]
                #stride_x_param = stride_x_param_array[random.randint(0,len(stride_x_param_array)-1)]
                if kernel_x_param == 6:
                    stride_x_param = stride_x_param_array[random.randint(0,0)]
                elif kernel_x_param == 10 or kernel_x_param == 15 or kernel_x_param == 20:
                    stride_x_param = stride_x_param_array[random.randint(0,1)]
                else:
                    stride_x_param = stride_x_param_array[random.randint(0,len(stride_x_param_array)-1)]
                # stride_y_param = stride_y_param_array[random.randint(0,len(stride_y_param_array))]
                stride_y_param = random.randint(1,2)
                train_with_parameter(dev, agent, max_epoch, use_cpu, batch_size, lr_param, decay_param, momentum_param, kernel_y_param, kernel_x_param, stride_y_param, stride_x_param, all_feature, all_label)

                
def train_with_parameter(dev, agent, max_epoch, use_cpu, batch_size, lr_param, decay_param, momentum_param, kernel_y_param, kernel_x_param, stride_y_param, stride_x_param, all_feature, all_label):
    opt = optimizer.SGD(momentum=momentum_param, weight_decay=decay_param)
    agent.push(MsgType.kStatus, 'Downlaoding data...')
    agent.push(MsgType.kStatus, 'Finish downloading data')
    n_folds = 5
    all_fold_test_auc = []
    for i, (train_index, test_index) in enumerate(StratifiedKFold(all_label.reshape(all_label.shape[0]), n_folds=n_folds)):
        test_auc_list = []
        train_feature, train_label, test_feature, test_label = all_feature[train_index], all_label[train_index], all_feature[test_index], all_label[test_index]
        print "fold: ", i
        # if i == 4:
        #    print "fold: ", i
        #    break
        in_shape = np.array([1, 12, 375])
        trainx = tensor.Tensor((batch_size, in_shape[0], in_shape[1], in_shape[2]), dev)
        trainy = tensor.Tensor((batch_size, ), dev, core_pb2.kInt)
        testx = tensor.Tensor((test_feature.shape[0], in_shape[0], in_shape[1], in_shape[2]), dev)
        testy = tensor.Tensor((test_feature.shape[0], ), dev, core_pb2.kInt)
        num_train_batch = train_feature.shape[0] / batch_size
        idx = np.arange(train_feature.shape[0], dtype=np.int32)

        hyperpara = np.array([12, 375, kernel_y_param, kernel_x_param, stride_y_param, stride_x_param])
        height, width, kernel_y, kernel_x, stride_y, stride_x = hyperpara[0], hyperpara[1], hyperpara[2], hyperpara[3], hyperpara[4], hyperpara[5]
        net = model.create_net(in_shape, hyperpara, use_cpu)
        net.to_device(dev)
    
        test_epoch = 5
        occlude_test_epoch = 100
        for epoch in range(max_epoch):
            if handle_cmd(agent):
                break
            np.random.seed(10)
            np.random.shuffle(idx)
            train_feature, train_label = train_feature[idx], train_label[idx]
            # print 'Epoch %d' % epoch
            
            loss, acc = 0.0, 0.0
            for b in range(num_train_batch):
                x, y = train_feature[b * batch_size:(b + 1) * batch_size], train_label[b * batch_size:(b + 1) * batch_size]
                x = x.reshape((batch_size, in_shape[0], in_shape[1], in_shape[2]))
                trainx.copy_from_numpy(x)
                trainy.copy_from_numpy(y)
                grads, (l, a), probs = net.train(trainx, trainy)
                loss += l
                acc += a
                for (s, p, g) in zip(net.param_specs(),
                                     net.param_values(), grads):
                    opt.apply_with_lr(epoch, lr_param, g, p, str(s.name))
                info = 'training loss = %f, training accuracy = %f' % (l, a)
                # utils.update_progress(b * 1.0 / num_train_batch, info)
            # put training status info into a shared queue
            info = dict(phase='train', step=epoch,
                        accuracy=acc/num_train_batch,
                        loss=loss/num_train_batch,
                        timestamp=time.time())
            agent.push(MsgType.kInfoMetric, info)
            info = 'training loss = %f, training accuracy = %f' \
                % (loss / num_train_batch, acc / num_train_batch)
            # print info
            
            if epoch % test_epoch == 0 or epoch == (max_epoch-1):
                loss, acc = 0.0, 0.0
                x, y = np.copy(test_feature), np.copy(test_label)
                x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
                testx.copy_from_numpy(x)
                testy.copy_from_numpy(y)
                l, a, probs = net.evaluate(testx, testy)
                loss += l
                acc += a
                print 'Epoch %d' % epoch
                print 'testing loss = %f, accuracy = %f' % (loss, acc)
                # put test status info into a shared queue
                info = dict(
                    phase='test',
                    step = epoch,
                    accuracy = acc,
                    loss = loss,
                    timestamp = time.time())
                agent.push(MsgType.kInfoMetric, info)
                test_auc = auroc(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
                print 'self calculate test auc = %f' % test_auc
                test_auc_list.append(test_auc) 
                # print 'self calculate test accuracy = %f' % cal_accuracy(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
                if epoch == (max_epoch-1):
                    np.savetxt('readmitted_prob.csv', softmax(tensor.to_numpy(probs))[:,1], fmt = '%6f', delimiter=",")
        net.save('parameter_last')
        all_fold_test_auc.append(np.asarray(test_auc_list).max())
    print "all_fold_test_auc: ", all_fold_test_auc
    print "all_fold_test_auc mean: ", np.asarray(all_fold_test_auc).mean()
    f = open('result-2.txt', 'a')
    f.write("new model result: " + "\n")
    f.write("lr_param: " + str(lr_param) + "\n")
    f.write("decay_param: " + str(decay_param) + "\n")
    f.write("momentum_param: " + str(momentum_param) + "\n")
    f.write("kernel_y_param: " + str(kernel_y_param) + "\n")
    f.write("kernel_x_param: " + str(kernel_x_param) + "\n")
    f.write("stride_y_param: " + str(stride_y_param) + "\n")
    f.write("stride_x_param: " + str(stride_x_param) + "\n")
    f.write("all_fold_test_auc mean: " + str(np.asarray(all_fold_test_auc).mean()) + "\n")
    f.write("\n")
        # print "begin explain"
        # explain_occlude_area(np.copy(test_feature), np.copy(test_label), 'readmitted_prob.csv', 'true_label_prob_matrix.csv', 'meta_data.csv', top_n = 20)
        # print "begin explain format out"
        # explain_occlude_area_format_out(np.copy(test_feature), np.copy(test_label), 'readmitted_prob.csv', 'true_label_prob_matrix.csv', 'meta_data.csv', top_n = 20)


 


if __name__ == '__main__':
    main()
