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
from singa import singa_wrap as singa
from singa.proto import model_pb2
from singa import tensor
from singa import loss
import numpy as np

class MultiSigmoidCrossEntropy(loss.Loss):
    '''This loss evaluates the cross-entropy loss between the prediction and the
    truth values with the prediction probability generated from Sigmoid.
    '''
    def __init__(self, epsilon=1e-8):
        super(SigmoidCrossEntropy, self).__init__()
        self.truth = None
        self.prob = None
        self.epsilon = epsilon  # to avoid log(x) with x being too small

    def forward(self, flag, x, y):
        '''loss is -yi * log pi - (1-yi) log (1-pi), where pi=sigmoid(xi)
        Args:
            flag (bool): true for training; false for evaluation
            x (Tensor): the prediction Tensor
            y (Tensor): the truth Tensor, a binary array value per sample
        Returns:
            a Tensor with one error value per sample
        '''
        p = tensor.sigmoid(x)
        if flag:
            self.truth = y
            self.prob = p
        np = 1 - p
        p += (p < self.epsilon) * self.epsilon
        np += (np < self.epsilon) * self.epsilon
        l = (y-1) * tensor.log(np) - y * tensor.log(p)
        # TODO(wangwei): add unary operation -Tensor
        return tensor.average(l, axis=1)

    def backward(self):
        ''' Compute the gradient of loss w.r.t to x.
        Returns:
            dx = pi - yi.
        '''
        assert self.truth is not None, 'must call forward in a prior'
        dx = self.prob - self.truth
        self.truth = None
        print 'backward dx shape: ', dx.shape
        return dx

    def evaluate(self, flag, x, y):
        '''Compuate the averaged error.
        Returns:
            a float value as the averaged error
        '''
        l = self.forward(False, x, y)
        return l.l1()