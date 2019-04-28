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
from sklearn.preprocessing import MinMaxScaler
from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType
from data_loader import *
from explain_occlude_area import *
from explain_occlude_area_format_out import *
from insert_into_table import *
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

def scale_prob(probpath, probrangepath):
    """scale probability values into (0, 1)"""
    probmatrix = np.genfromtxt(probpath, delimiter=",", dtype=str)
    prob = probmatrix[:, 1].astype(np.float32)
    min_value, max_value = min(prob), max(prob)
    prob_range = np.genfromtxt(probrangepath, delimiter=",", dtype=np.float32)
    old_min, old_max = prob_range
    min_value = min(min_value, old_min)
    max_value = max(max_value, old_max)    
    # update new prob range
    print 'prob range: [%.6f, %.6f]' % (min_value, max_value)
    np.savetxt(probrangepath, np.array([min_value, max_value]), fmt='%6f', delimiter=",")
    scaled_values = (prob - min_value) / (max_value - min_value)
    scaled_values = scaled_values.astype(str).reshape(-1, 1)
    probability = np.concatenate((probmatrix, scaled_values), axis=1)
    np.savetxt(probpath, probability, fmt = '%s,%s,%s', delimiter=",")
                

def train(inputfolder, outputfolder, visfolder, sampleid, dev, agent, max_epoch, use_cpu, batch_size=100):
    opt = optimizer.SGD(momentum=0.8, weight_decay=0.01)
    agent.push(MsgType.kStatus, 'Downlaoding data...')
    start = time.time()
    print 'start loading data...'
    all_feature, all_patients = get_data(os.path.join(inputfolder, 'deployment_features.txt'), None, os.path.join(inputfolder, 'deployment_patients_id.txt'), True)  # PUT THE DATA on/to dbsystem
    print 'loading time spent = ', time.time()-start
    agent.push(MsgType.kStatus, 'Finish downloading data')
    n_folds = 5
    test_feature, test_patient_ids = all_feature, all_patients
    dim = int(test_feature.shape[1]/12)
    in_shape = [1, 12, dim]
    testx = tensor.Tensor((batch_size, in_shape[0], in_shape[1], in_shape[2]), dev)
    print 'test_feature.shape = ', test_feature.shape, 'test_patient_ids.shape = ', test_patient_ids.shape
    num_test_batch = int(test_feature.shape[0] / batch_size)

    for kernel_y, stride_y in zip([2], [1]):
        for kernel_x in [10]:
            for stride_x in [3]:
                hyperpara = np.array([12, dim, kernel_y, kernel_x, stride_y, stride_x])
                height, width, kernel_y, kernel_x, stride_y, stride_x = hyperpara[0], hyperpara[1], hyperpara[2], hyperpara[3], hyperpara[4], hyperpara[5]
                check_file = '_'.join(['new_parameter', str(kernel_y), str(kernel_x), str(stride_y), str(stride_x)])
                print "checkpoint path: ", check_file
                check_file = os.path.join('checkpoints', check_file+'.pickle')
                if not os.path.exists(check_file):
                    print 'checkpoint file %s does not exist. ' % check_file
                    continue
                net = model.create_net(in_shape, hyperpara, use_cpu)
                net.load(check_file, 20, True)
                net.to_device(dev)
                for name in zip(net.param_names()):
                    print "init names: ", name
                
                if handle_cmd(agent):
                    break
                np.random.seed(10)
                probability = np.zeros((batch_size, 2))
                patients = 0
                for b in range(num_test_batch):
                    x = test_feature[b * batch_size: (b+1)* batch_size]
                    patient_id = test_patient_ids[b * batch_size: (b+1)* batch_size]
                    x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
                    testx.copy_from_numpy(x)
                    probs = tensor.to_numpy(net.predict_prob(testx))
                    probs = softmax(probs)
                    if b == 0:
                        probability = probs 
                        patients = patient_id
                    else:
                        probability = np.concatenate((probability, probs), axis=0)
                        patients = np.concatenate((patients, patient_id), axis=0)
                x = test_feature[b * batch_size: ]
                patient_id = test_patient_ids[b * batch_size:]
                x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
                testx = tensor.Tensor((x.shape[0], in_shape[0], in_shape[1], in_shape[2]), dev)
                testx.copy_from_numpy(x)
                probs = tensor.to_numpy(net.predict_prob(testx))
                probs = softmax(probs)
                if isinstance(patients, int):
                    probability = probs 
                    patients = patient_id
                else:
                    probability = np.concatenate((probability, probs), axis=0)
                    patients = np.concatenate((patients, patient_id), axis=0)
                print 'probability.shape = ', probability.shape 
                probpath = os.path.join(outputfolder, 'readmitted_prob.csv')
                probability = softmax(probability)[:, 1]
                np.savetxt(probpath, np.transpose((patients, probability)), fmt = '%s,%s', delimiter=",")
                
                probrangepath = os.path.join(outputfolder, 'readmitted_prob_range.csv')
                prob_range = np.array([min(probability), max(probability)])
                np.savetxt(probrangepath, prob_range, fmt='%6f', delimiter=",")

                # put test status info into a shared queue
                info = dict(
                    phase='test',
                    # step = epoch,
                    # accuracy = acc,
                    # loss = loss,
                    timestamp = time.time())
                agent.push(MsgType.kInfoMetric, info)
                
                metadatapath = os.path.join(outputfolder, 'meta_data.csv')
                truelabelprobpath = os.path.join(outputfolder, 'true_label_prob_matrix.csv')
                patientdrgpath = os.path.join(inputfolder, 'patient_DRG_info.pkl')
                drgpath = os.path.join(inputfolder, 'DRG_dict.pkl')
                
                height_dim = (height - kernel_y) / stride_y + 1; 
                width_dim = (width - kernel_x) / stride_x + 1;
                meta_data = np.array([height_dim, height, kernel_y, stride_y, width_dim, width, kernel_x, stride_x])
                np.savetxt(metadatapath, meta_data, fmt = '%6f', delimiter=",") #modify here
            
                # print "occlude test"
                # # occlude test data
                # true_label_prob_matrix = np.zeros([(height_dim * width_dim), 1])
                # for height_idx in range(height_dim):
                #     for width_idx in range(width_dim):
                #         occlude_test_feature, occlude_test_label = get_occlude_data(np.copy(test_feature), np.copy(test_label), \
                #         height, width, height_idx, width_idx, kernel_y, kernel_x, stride_y, stride_x)
                #         loss, acc = 0.0, 0.0
                #         x, y = occlude_test_feature, occlude_test_label # !!! where are the labels?
                #         x = x.reshape((x.shape[0], in_shape[0], in_shape[1], in_shape[2]))
                #         testx.copy_from_numpy(x)
                #         testy.copy_from_numpy(y)
                #         l, a, probs = net.evaluate(testx, testy)
                #         y_scores = softmax(tensor.to_numpy(probs))[:,1]
                #         sum_true_label_prob = 0.0
                #         for i in range(0, x.shape[0]): # !!! y_scores ~~ the probability of 1 !!!
                #             if y[i] == 1:
                #                 sum_true_label_prob = sum_true_label_prob + y_scores[i]
                #             elif y[i] == 0:
                #                 sum_true_label_prob = sum_true_label_prob + (1 - y_scores[i])
                #         true_label_prob_matrix[height_idx * width_dim + width_idx, 0] = sum_true_label_prob / x.shape[0]
                # print "occlude x shape: ", x.shape
                # np.savetxt(truelabelprobpath, true_label_prob_matrix, fmt = '%6f', delimiter=",") #modify here

                # for (s, p) in zip(net.param_specs(), net.param_values()):
                #     print "last epoch param name: ", s
                #     print "last epoch param value: ", p.l2()
                # print "begin explain"

                # explain_occlude_area(np.copy(test_feature), None, probpath, truelabelprobpath, metadatapath, top_n = 20)
                # print "begin explain format out"
                top_n = 30
                print "top_n: ", top_n
                # explain_occlude_area_format_out(visfolder, np.copy(test_feature), None, probpath, truelabelprobpath, metadatapath, patientdrgpath, drgpath, top_n = top_n)
                
                scale_prob(probpath, probrangepath)
                insert_into_table(np.copy(test_feature), probpath, truelabelprobpath, metadatapath, patientdrgpath, drgpath, top_n, 'readmin', 'Zhaojing2018!', '172.16.199.1', 'discovery_web', 'result_prediction')
                # insert_into_table(np.copy(test_feature), probpath, truelabelprobpath, metadatapath, patientdrgpath, drgpath, top_n, 'root', '@42cNa4h', 'localhost', 'test_nuh', 'readm_pred')

if __name__ == '__main__':
    main()
