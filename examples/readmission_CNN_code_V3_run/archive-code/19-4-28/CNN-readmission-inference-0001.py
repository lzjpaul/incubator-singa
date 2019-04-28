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
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, roc_auc_score

from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType
from data_loader import *
from explain_occlude_area import *
from explain_occlude_area_format_out import *
from healthcare_metrics import *
import model
import json
import pickle as cPickle
import pdb
import os


def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train CNN Readmission Model")
        parser.add_argument('-inputfolder', type=str, help='inputfolder')
        parser.add_argument('-outputfolder', type=str, help='outputfolder')
        parser.add_argument('-visfolder', type=str, help='visfolder')
        parser.add_argument('-sampleid', type=int, help='the sample id for output')
        parser.add_argument('-p', '--port', default=9989, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=10)

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
        train(args.inputfolder, args.outputfolder, args.visfolder, args.sampleid, dev, agent, args.max_epoch, use_cpu)
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

def train(inputfolder, outputfolder, visfolder, sampleid, dev, agent, max_epoch, use_cpu, batch_size=100):
    opt = optimizer.SGD(momentum=0.8, weight_decay=0.01)
    agent.push(MsgType.kStatus, 'Downlaoding data...')
    start = time.time()
    print 'start loading data...'
    all_feature, all_label, all_patients = get_data(os.path.join(inputfolder, 'test_features.npz'), os.path.join(inputfolder, 'test_label.txt'), os.path.join(inputfolder, 'test_patient_ids.txt'))  # PUT THE DATA on/to dbsystem
    print 'loading time spent = ', time.time()-start
    agent.push(MsgType.kStatus, 'Finish downloading data')
    all_label = all_label[:,1]
    test_feature, test_label, test_patient_ids = all_feature, all_label, all_patients
    print 'test_feature.shape = ', test_feature.shape, ', test_label.shape = ', test_label.shape, 'test_patient_ids.shape = ', test_patient_ids.shape
    
    print "test label sum: ", test_label.sum()
    dim = int(test_feature.shape[1]/12)
    in_shape = [1, 12, dim]
    testx = tensor.Tensor((batch_size, in_shape[0], in_shape[1], in_shape[2]), dev)
    testy = tensor.Tensor((batch_size, ), dev, core_pb2.kInt)
    print 'testx.shape = ', testx.shape
    print 'testy.shape = ', testy.shape
    num_test_batch = int(test_feature.shape[0] / batch_size)

    # height = 12
    # width = 375
    # kernel_y = 3
    # kernel_x = 80
    # stride_y = 1
    # stride_x = 20
    # hyperpara = np.array([12, dim, 3, 10, 1, 3])
    threshold = 1000.0
    variances = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    for kernel_y, stride_y in zip([2], [1]):
        for kernel_x in [80]:
            for stride_x in [10]:
                hyperpara = np.array([12, dim, kernel_y, kernel_x, stride_y, stride_x])

                height, width, kernel_y, kernel_x, stride_y, stride_x = hyperpara[0], hyperpara[1], hyperpara[2], hyperpara[3], hyperpara[4], hyperpara[5]
                for var in variances: 
                    net = model.create_net(in_shape, hyperpara, use_cpu)
                    check_file = '_'.join(['new_parameter', str(kernel_y), str(kernel_x), str(stride_y), str(stride_x), str(threshold), str(var), '00001'])
                    check_file = os.path.join('checkpoints', check_file + '.pickle')
                    print "checkpoint path: ", check_file
                    if not os.path.exists(check_file):
                        continue
                    net.load(check_file, 20, True)
                    net.to_device(dev)
                    for name in zip(net.param_names()):
                        print "init names: ", name
                    
                    if handle_cmd(agent):
                        break
                    np.random.seed(10)
                    loss, acc = 0.0, 0.0
                    probability = np.zeros((batch_size, 2))
                    patients = []
                    ytrue = np.zeros((batch_size, 2))
                    for b in range(num_test_batch):
                        x, y = test_feature[b * batch_size: (b+1)* batch_size], test_label[b * batch_size:(b + 1) * batch_size]
                        patient_id = test_patient_ids[b * batch_size: (b+1)* batch_size]
                        # x, y = np.copy(test_feature), np.copy(test_label)
                        x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
                        testx.copy_from_numpy(x)
                        testy.copy_from_numpy(y)
                        l, a, probs = net.evaluate(testx, testy)
                        if b == 0:
                            probability = tensor.to_numpy(probs)
                            ytrue = y
                            patients = patient_id
                        else:
                            probability = np.concatenate((probability, tensor.to_numpy(probs)), axis=0)
                            ytrue = np.concatenate((ytrue, y), axis=0)
                            patients = np.concatenate((patients, patient_id), axis=0)
                        loss += l
                        acc += a

                    loss /= num_test_batch
                    acc /= num_test_batch
                    print 'testing loss = %f, accuracy = %f' % (loss, acc)
                    # put test status info into a shared queue
                    info = dict(
                        phase='test',
                        # step = epoch,
                        accuracy = acc,
                        loss = loss,
                        timestamp = time.time())
                    agent.push(MsgType.kInfoMetric, info)
                    # print 'self calculate test auc = %f' % auroc(softmax(probability)[:,1].reshape(-1, 1), ytrue.reshape(-1, 1))
                    print 'self calculate test accuracy = %f' % cal_accuracy(softmax(probability)[:,1].reshape(-1, 1), ytrue.reshape(-1, 1))
                    cnn_metric_dict = {} # for output to json
                    cnn_metric_dict['Number of Samples: '] = ytrue.shape[0]
                    cnn_sensitivity, cnn_specificity, cnn_harmonic = HealthcareMetrics(softmax(probability)[:,1].reshape(-1, 1), ytrue.reshape(-1, 1), 0.25)
                    # cnn_metric_dict['AUC: '] = auroc(softmax(probability)[:,1].reshape(-1, 1), ytrue.reshape(-1, 1))
                    cnn_metric_dict['accuracy: '] = cal_accuracy(softmax(probability)[:,1].reshape(-1, 1), ytrue.reshape(-1, 1))
                    cnn_metric_dict['Sensitivity: '] = cnn_sensitivity
                    cnn_metric_dict['Specificity: '] = cnn_specificity

                    try:
                        with open(os.path.join(visfolder, 'cnn_metric_info.json'), 'a') as cnn_metric_info_writer:
                            # json.dump(cnn_metric_dict, cnn_metric_info_writer)
                            cnn_metric_info_writer.write('[')
                            # cnn_metric_info_writer.write('\"Number of Patients: %d\", '%(y.shape[0]))
                            cnn_metric_info_writer.write('\"Checkpoint file: %s\", ' % (check_file))
                            # cnn_metric_info_writer.write('\"AUC: %s\", '% (str(int(100 * round((auroc(softmax(probability)[:,1].reshape(-1, 1), ytrue.reshape(-1, 1))),2)))+'%') )
                            cnn_metric_info_writer.write('\"Accuracy: %s\", ' % (str(int(100 * cnn_metric_dict['accuracy: '])) + '%'))
                            cnn_metric_info_writer.write('\"Sensitivity: %s\", '%(str(int(100 * round(cnn_sensitivity,2)))+'%'))
                            cnn_metric_info_writer.write('\"Specificity: %s\" '%(str(int(100* round(cnn_specificity,2)))+'%'))
                            cnn_metric_info_writer.write(']\n')
                    except Exception as e:
                        os.remove(os.path.join(visfolder, 'cnn_metric_info.json'))
                        print('output cnn_metric_info.json failed: ', e)
                    
                    # save predicted readimission probability
                    file_s = "_".join(['readmitted_prob', str(kernel_y), str(kernel_x), str(stride_y), str(stride_x), str(threshold), str(var), '00001'])
                    file_s += '.csv'
                    probpath = os.path.join(outputfolder, file_s)
                    np.savetxt(probpath, np.transpose((softmax(probability)[:,1], ytrue)), fmt = '%6f, %6f', delimiter=",")
                    # np.savetxt(probpath, np.transpose((patients, softmax(probability)[:,1])), fmt = '%s,%s', delimiter=",")
                    

                # truelabelprobpath = os.path.join(outputfolder,'true_label_prob_matrix.csv')
                # metadatapath = os.path.join(outputfolder,'meta_data.csv')
                # patientdrgpath = os.path.join(inputfolder, 'patient_DRG_info.pkl')
                # drgpath = os.path.join(inputfolder, 'DRG_dict.pkl')
                # # # occlude test data
                # # print "occlude test"
                # # height_dim = (height - kernel_y) / stride_y + 1; 
                # # width_dim = (width - kernel_x) / stride_x + 1;
                # # meta_data = np.array([height_dim, height, kernel_y, stride_y, width_dim, width, kernel_x, stride_x])
                # # np.savetxt(metadatapath, meta_data, fmt = '%6f', delimiter=",") #modify here
                # # true_label_prob_matrix = np.zeros([(height_dim * width_dim), 1])
                # # for height_idx in range(height_dim):
                # #     for width_idx in range(width_dim):
                # #         sum_true_label_prob = 0.0
                # #         for b in range(num_test_batch):
                # #             x, y = test_feature[b * batch_size: (b+1)* batch_size], test_label[b * batch_size:(b + 1) * batch_size]
                # #             occlude_test_feature, occlude_test_label = get_occlude_data(np.copy(x), np.copy(y), \
                # #             height, width, height_idx, width_idx, kernel_y, kernel_x, stride_y, stride_x)
                # #             loss, acc = 0.0, 0.0
                # #             x, y = occlude_test_feature, occlude_test_label # !!! where are the labels?
                # #             x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
                # #             testx.copy_from_numpy(x)
                # #             testy.copy_from_numpy(y)
                # #             l, a, probs = net.evaluate(testx, testy)
                # #             y_scores = softmax(tensor.to_numpy(probs))[:,1]
                # #             for i in range(0, x.shape[0]): # !!! y_scores ~~ the probability of 1 !!!
                # #                 if y[i] == 1:
                # #                     sum_true_label_prob += y_scores[i]
                # #                 elif y[i] == 0:
                # #                     sum_true_label_prob += (1 - y_scores[i])
                # #         true_label_prob_matrix[height_idx * width_dim + width_idx, 0] += sum_true_label_prob 
                # # true_label_prob_matrix /= int(num_test_batch*batch_size)
                # # print "occlude x shape: ", x.shape
                # # np.savetxt(truelabelprobpath, true_label_prob_matrix, fmt = '%6f', delimiter=",") #modify here

                # # for (s, p) in zip(net.param_specs(), net.param_values()):
                # #     print "last epoch param name: ", s
                # #     print "last epoch param value: ", p.l2()

                # # print "begin explain"
                # # explain_occlude_area(np.copy(test_feature[:10]), np.copy(test_label[:10]), probpath, truelabelprobpath, metadatapath, top_n = 20)
                
                # # # print "begin explain format out"
                # top_n = 30
                # print "top_n: ", top_n
                # for b in range(num_test_batch):
                #     x, y = test_feature[b * batch_size: (b+1)* batch_size], test_label[b * batch_size:(b + 1) * batch_size]
                #     explain_occlude_area_format_out(visfolder, np.copy(x), np.copy(y), probpath, truelabelprobpath, metadatapath, patientdrgpath, drgpath, top_n = top_n)
                #     pdb.set_trace()
 
if __name__ == '__main__':
    main()
