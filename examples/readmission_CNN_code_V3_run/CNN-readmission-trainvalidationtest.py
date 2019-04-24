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

# 11-12
# no testing data (no read in, also no test_epoch)

# references:
# https://github.com/lzjpaul/incubator-singa/blob/CNN-readmission/examples/CNN-readmission/train-5-fold.py

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
from explain_occlude_area import *
from explain_occlude_area_format_out import *
import model
import cPickle
import param_process 
import pdb

def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train CNN Readmission Model")
        parser.add_argument('-inputfolder', type=str, help='inputfolder')
        parser.add_argument('-outputfolder', type=str, help='outputfolder')
        parser.add_argument('-visfolder', type=str, help='visfolder')
        parser.add_argument('-trainratio', type=float, help='ratio of train samples')
        parser.add_argument('-validationratio', type=float, help='ratio of validation samples')
        parser.add_argument('-testratio', type=float, help='ratio of test samples')        
        parser.add_argument('-p', '--port', default=9989, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=30)

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
        train(args.inputfolder, args.outputfolder, args.visfolder, args.trainratio, args.validationratio, args.testratio, dev, agent, int(args.max_epoch), use_cpu)
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

# def auroc(yPredictProba, yTrue):
#     return roc_auc_score(yTrue, yPredictProba)


def train(inputfolder, outputfolder, visfolder, trainratio, validationratio, testratio, dev, agent, max_epoch, use_cpu, batch_size=100):
    opt = optimizer.SGD()
    agent.push(MsgType.kStatus, 'Downlaoding data...')
    start = time.time()
    print "max_epoch: ", max_epoch
    print 'start loading data...'
    all_feature, all_label, _ = get_data(os.path.join(inputfolder, 'train_features.npz'), os.path.join(inputfolder, 'train_label.txt'), os.path.join(inputfolder, 'train_patient_ids.txt'))  # PUT THE DATA on/to dbsystem
    # all_feature, all_label, _ = get_data(os.path.join(inputfolder, 'test_features.npz'), os.path.join(inputfolder, 'test_label.txt'), os.path.join(inputfolder, 'test_patient_ids.txt'))  
    print 'loading time spent = ', time.time()-start
    agent.push(MsgType.kStatus, 'Finish downloading data')
    n_folds = 5
    print "all_label shape: ", all_label.shape
    all_label = all_label[:,1]
    train_feature, train_label = all_feature, all_label
    print 'train_feature.shape = ', train_feature.shape, ', train_label.shape = ', train_label.shape
    print "train label sum: ", train_label.sum()
    dim = int(all_feature.shape[1]/12)
    in_shape = np.array([1, 12, dim])
    trainx = tensor.Tensor((batch_size, in_shape[0], in_shape[1], in_shape[2]), dev)
    trainy = tensor.Tensor((batch_size, ), dev, core_pb2.kInt)
    num_train = train_feature.shape[0]
    num_train_batch = num_train / batch_size
    idx = np.arange(num_train, dtype=np.int32)

    # height = 12
    # width = 375
    # kernel_y = 3
    # kernel_x = 80
    # stride_y = 1
    # stride_x = 20
    coefficient = 0.01
    threshold = 4.0
    variances = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    for kernel_y, stride_y in zip([2], [1]):
        for kernel_x in [80]:
            for stride_x in [10]:
                hyperpara = np.array([12, dim, kernel_y, kernel_x, stride_y, stride_x])
                height, width, kernel_y, kernel_x, stride_y, stride_x = hyperpara[0], hyperpara[1], hyperpara[2], hyperpara[3], hyperpara[4], hyperpara[5]
                print 'kernel_y: ', kernel_y
                print 'kernel_x: ', kernel_x
                print 'stride_y: ', stride_y
                print 'stride_x: ', stride_x
                for var in variances:
                    net = model.create_net(in_shape, hyperpara, use_cpu)
                    net.to_device(dev)
                    param_pro = param_process.param_process(0, var)

                    test_epoch = 10
                    occlude_test_epoch = 100
                    for epoch in range(max_epoch):
                        if handle_cmd(agent):
                            break
                        # np.random.seed(10)
                        # np.random.shuffle(idx)
                        # train_feature, train_label = train_feature[idx], train_label[idx]
                        # actually it is epoch step
                        print 'Epoch: %d' % epoch
                        
                        loss, acc = 0.0, 0.0
                        for b in range(num_train_batch):
                            sample_index = np.random.choice(num_train, size=batch_size, replace=True)
                            # x, y = train_feature[b * batch_size:(b + 1) * batch_size], train_label[b * batch_size:(b + 1) * batch_size]
                            x, y = train_feature[sample_index], train_label[sample_index]
                            x = x.reshape((batch_size, in_shape[0], in_shape[1], in_shape[2]))
                            trainx.copy_from_numpy(x)
                            trainy.copy_from_numpy(y)
                            grads, (l, a), probs = net.train(trainx, trainy)
                            loss += l
                            acc += a
                            conv1_grad_list=[]
                            dense_grad_list=[]
                            for (s, p, g) in zip(net.param_specs(),
                                                 net.param_values(), grads):
                                # print 'name: ', str(s.name)
                                # print 'param l2: ', p.l2()
                                # print '1.0 * grad l2: ', g.l2()
                                if 'conv1' in str(s.name):
                                    conv1_grad_list.append(tensor.to_numpy(g))
                                else:
                                    dense_grad_list.append(tensor.to_numpy(g))
                            print 'len(conv1_grad_list): ', len(conv1_grad_list)
                            print 'len(dense_grad_list): ', len(dense_grad_list)
                            clip_coefficient_dict = param_pro.calculate_clip_coefficient(conv1_grad_list, dense_grad_list, threshold)
                            for (s, p, g) in zip(net.param_specs(),
                                                 net.param_values(), grads):
                                # print 'name: ', str(s.name)
                                # print 'param l2: ', p.l2()
                                # print '1.0 * grad l2: ', g.l2()
                                g = param_pro.apply_with_regularizer_constraint_noise(coefficient, str(s.name), p, g, clip_coefficient_dict, dev)
                                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s.name))
                            
                            info = 'step = %d, step training loss = %f, step training accuracy = %f' % (b, l, a)
                            if b % 100 == 0:
                                print info
                        # utils.update_progress(b * 1.0 / num_train_batch, info)
                        # utils.update_progress(epoch * 1.0 / max_epoch, info)

                        # put training status info into a shared queue
                        info = dict(phase='train', step=epoch,
                                    accuracy=acc/num_train_batch,
                                    loss=loss/num_train_batch,
                                    timestamp=time.time())
                        agent.push(MsgType.kInfoMetric, info)
                        info = 'epoch = %d, epoch avg training loss (important) = %f, epoch avg training accuracy = %f' \
                            % (epoch, loss / num_train_batch, acc / num_train_batch)
                        print info
                      
                    # pdb.set_trace() 
                    check_file = '_'.join(['new_parameter', str(kernel_y), str(kernel_x), str(stride_y), str(stride_x), str(threshold), str(var)])
                    check_file = os.path.join('checkpoints', check_file)
                    print 'checkpoint_file = ', check_file
                    net.save(check_file, 20, True)     
    # print "begin explain"
    # explain_occlude_area(np.copy(test_feature), np.copy(test_label), 'readmitted_prob.csv', 'true_label_prob_matrix.csv', 'meta_data.csv', top_n = 20)
    # print "begin explain format out"
    # explain_occlude_area_format_out(np.copy(test_feature), np.copy(test_label), 'readmitted_prob.csv', 'true_label_prob_matrix.csv', 'meta_data.csv', top_n = 20)


 


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=1 python CNN-readmission-trainvalidationtest.py -inputfolder ../saved_sparse_data/ -outputfolder 'outputfolder' -visfolder 'visfolder' --max_epoch 50
# CUDA_VISIBLE_DEVICES=0 python CNN-readmission-trainvalidationtest.py -inputfolder ../saved_sparse_data/ -outputfolder 'outputfolder' -visfolder 'visfolder'
