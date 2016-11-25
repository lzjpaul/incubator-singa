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
import os, sys, shutil
import urllib
import cPickle
import numpy as np

data_file = '/data/zhaojing/regularization/CMSHF/CMS_HF_VECTOR_Regulariz_diag_features.txt'
label_file = '/data/zhaojing/regularization/CMSHF/CMS_HF_VECTOR_Regulariz_label.csv'
correl_file = '/data/zhaojing/regularization/CMSHF/CMSHFSimilarityMatrix2level.txt' 


def load_dataset(train_num):
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