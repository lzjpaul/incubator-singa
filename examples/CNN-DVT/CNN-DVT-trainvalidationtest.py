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

# need to modify max-epoch, lr, decay and momentum also
import sys, os
import traceback
import time
import urllib
import numpy as np
from argparse import ArgumentParser
# from sklearn.cross_validation import StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, roc_auc_score

from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType
from data_loader import *
# from explain_occlude_area import *
# from explain_occlude_area_format_out import *
import model
from healthcare_metrics import *


def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train CNN Readmission Model")
        parser.add_argument('-inputfolder', type=str, help='inputfolder')
        parser.add_argument('-outputfolder', type=str, help='outputfolder')      
        parser.add_argument('-p', '--port', default=9989, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=400)

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
        train(args.inputfolder, args.outputfolder, dev, agent, args.max_epoch, get_lr, use_cpu)
        # wait the agent finish handling http request
        agent.stop()
    except SystemExit:
        return
    except:
        # p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")

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
    return 0.03


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

def cal_accuracy(yPredict, yTrue):
    return np.sum(((yPredict > 0.5) == yTrue).astype(int)) / float(yTrue.shape[0])

# def auroc(yPredictProba, yTrue):
#     return roc_auc_score(yTrue, yPredictProba)

def train(inputfolder, outputfolder, dev, agent, max_epoch, get_lr, use_cpu, batch_size=100):
    opt = optimizer.SGD(momentum=0.95, weight_decay=0.01)
    agent.push(MsgType.kStatus, 'Downlaoding data...')
    # all_feature, all_label = get_data(os.path.join(inputfolder, 'features.txt'), os.path.join(inputfolder, 'label.txt'))  # PUT THE DATA on/to dbsystem
    train_feature, train_label = get_data(os.path.join(inputfolder, 'DVT_diag_labtest_train_data.csv'), os.path.join(inputfolder, 'DVT_diag_labtest_train_label.csv'))  # PUT THE DATA on/to dbsystem
    valid_feature, valid_label = get_data(os.path.join(inputfolder, 'DVT_diag_labtest_valid_data.csv'), os.path.join(inputfolder, 'DVT_diag_labtest_valid_label.csv'))  # PUT THE DATA on/to dbsystem
    test_feature, test_label = get_data(os.path.join(inputfolder, 'DVT_diag_labtest_test_data.csv'), os.path.join(inputfolder, 'DVT_diag_labtest_test_label.csv'))  # PUT THE DATA on/to dbsystem
    agent.push(MsgType.kStatus, 'Finish downloading data')
    print "train_feature shape: ", train_feature.shape
    print "train_label shape: ", train_label.shape
    print "valid_feature shape: ", valid_feature.shape
    print "valid_label shape: ", valid_label.shape
    print "test_feature shape: ", test_feature.shape
    print "test_label shape: ", test_label.shape
    # all_label = all_label[:,1]
    print "train label sum: ", train_label.sum()
    print "test label sum: ", test_label.sum()
    in_shape = np.array([1, 10, 268])
    trainx = tensor.Tensor((batch_size, in_shape[0], in_shape[1], in_shape[2]), dev)
    trainy = tensor.Tensor((batch_size, ), dev, core_pb2.kInt)
    validx = tensor.Tensor((valid_feature.shape[0], in_shape[0], in_shape[1], in_shape[2]), dev)
    validy = tensor.Tensor((valid_feature.shape[0], ), dev, core_pb2.kInt)
    testx = tensor.Tensor((test_feature.shape[0], in_shape[0], in_shape[1], in_shape[2]), dev)
    testy = tensor.Tensor((test_feature.shape[0], ), dev, core_pb2.kInt)
    num_train_batch = train_feature.shape[0] / batch_size
    # num_test_batch = test_x.shape[0] / (batch_size)
    idx = np.arange(train_feature.shape[0], dtype=np.int32)

    # height = 12
    # width = 375
    # kernel_y = 3
    # kernel_x = 80
    # stride_y = 1
    # stride_x = 20
    # hyperpara = np.array([10, 268, 3, 10, 1, 3])
    hyperpara = np.array([10, 268, 2, 2, 1, 1])
    height, width, kernel_y, kernel_x, stride_y, stride_x = hyperpara[0], hyperpara[1], hyperpara[2], hyperpara[3], hyperpara[4], hyperpara[5]
    print 'kernel_y: ', kernel_y
    print 'kernel_x: ', kernel_x
    print 'stride_y: ', stride_y
    print 'stride_x: ', stride_x
    net = model.create_net(in_shape, hyperpara, use_cpu)
    net.to_device(dev)
    
    valid_epoch = 10
    test_epoch = 10
    for epoch in range(max_epoch):
        if handle_cmd(agent):
            break
        np.random.seed(10)
        np.random.shuffle(idx)
        train_feature, train_label = train_feature[idx], train_label[idx]
        print 'Epoch %d' % epoch
        
        loss, acc = 0.0, 0.0
        for b in range(num_train_batch):
            x, y = train_feature[b * batch_size:(b + 1) * batch_size], train_label[b * batch_size:(b + 1) * batch_size]
            x = x.reshape((batch_size, in_shape[0], in_shape[1], in_shape[2]))
            trainx.copy_from_numpy(x)
            trainy.copy_from_numpy(y)
            grads, (l, a) = net.train(trainx, trainy)
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
        
        if epoch % valid_epoch == 0 or epoch == (max_epoch-1):
            loss, acc = 0.0, 0.0
            x, y = np.copy(valid_feature), np.copy(valid_label)
            x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
            validx.copy_from_numpy(x)
            validy.copy_from_numpy(y)
            l, a, probs = net.evaluate(validx, validy)
            loss += l
            acc += a
            print 'valid loss = %f, accuracy = %f' % (loss, acc)
            # put test status info into a shared queue
            info = dict(
                phase='valid',
                step = epoch,
                accuracy = acc,
                loss = loss,
                timestamp = time.time())
            agent.push(MsgType.kInfoMetric, info)
            # print 'self calculate valid auc = %f' % auroc(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            # print 'self calculate valid accuracy = %f' % cal_accuracy(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            dvt_precision_1, dvt_recall_1, dvt_F_measure_1 = HealthcareMetrics(probs.shape[0], softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1), 0.5)
            print 'valid dvt_precision_1 = %f, valid dvt_recall_1 = %f, valid dvt_F_measure_1 = %f' % (dvt_precision_1, dvt_recall_1, dvt_F_measure_1)

        if epoch % test_epoch == 0 or epoch == (max_epoch-1):
            loss, acc = 0.0, 0.0
            x, y = np.copy(test_feature), np.copy(test_label)
            x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
            testx.copy_from_numpy(x)
            testy.copy_from_numpy(y)
            l, a, probs = net.evaluate(testx, testy)
            loss += l
            acc += a
            print 'testing loss = %f, accuracy = %f' % (loss, acc)
            # put test status info into a shared queue
            info = dict(
                phase='test',
                step = epoch,
                accuracy = acc,
                loss = loss,
                timestamp = time.time())
            agent.push(MsgType.kInfoMetric, info)
            # print 'self calculate test auc = %f' % auroc(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            print 'self calculate test accuracy = %f' % cal_accuracy(softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1))
            dvt_precision_1, dvt_recall_1, dvt_F_measure_1 = HealthcareMetrics(probs.shape[0], softmax(tensor.to_numpy(probs))[:,1].reshape(-1, 1), y.reshape(-1, 1), 0.5)
            print 'test dvt_precision_1 = %f, test dvt_recall_1 = %f, test dvt_F_measure_1 = %f' % (dvt_precision_1, dvt_recall_1, dvt_F_measure_1)
            net.save(os.path.join(outputfolder,'parameter_last'+str(epoch)), 20)
        
    for (s, p) in zip(net.param_specs(), net.param_values()):
        print "last epoch param name: ", s
        print "last epoch param value: ", p.l2()
    print ('begin save params')
    net.save(os.path.join(outputfolder,'parameter_last'), 20)
    print ('end save params')
 


if __name__ == '__main__':
    main()
