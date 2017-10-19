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

from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType
from data_loader import *
from explain_occlude_area import *
from explain_occlude_area_format_out import *
import model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score


def get_occlude_data(occlude_feature, occlude_label, height, width, height_idx, width_idx, kernel_y, kernel_x, stride_y, stride_x):
    '''load occlude data'''
    for n in range(occlude_feature.shape[0]): #sample
        for j in range (kernel_y):
            occlude_feature[n, ((height_idx * stride_y + j) * width + width_idx * stride_x) : ((height_idx * stride_y + j) * width + width_idx * stride_x + kernel_x)] = float(0.0)
    return occlude_feature, occlude_label


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

def train():
    all_feature, all_label = get_data\
    ('/data/zhaojing/regularization/LACE-CNN-1500/reverse-order/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_12slots_reverse.csv', \
    '/data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv')  # PUT THE DATA on/to dbsystem
    n_folds = 5
    for i, (train_index, test_index) in enumerate(StratifiedKFold(all_label.reshape(all_label.shape[0]), n_folds=n_folds)):
        train_feature, train_label, test_feature, test_label = all_feature[train_index], all_label[train_index], all_feature[test_index], all_label[test_index]
        if i == 4:
            print "fold: ", i
            break
    
    # model = LogisticRegression(penalty = 'l1', tol=0.000001)
    model = LogisticRegression(penalty = 'l2')
    model.fit(train_feature, train_label)
    modelPred = model.predict_proba(test_feature)
    fpr, tpr, thresholds = metrics.roc_curve(test_label, modelPred[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print "auc = \n", auc

if __name__ == '__main__':
    train()
