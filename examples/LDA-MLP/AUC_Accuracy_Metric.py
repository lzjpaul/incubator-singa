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
from singa import tensor
import numpy as np
from singa import metric
from sklearn.metrics import roc_auc_score

class AUCAccuracy(metric.Metric):
    '''Make the top-k labels of max probability as the prediction
    Compute the precision against the groundtruth labels
    '''
    def forward(self, x, y):
        '''Compute the precision for each sample.
        Convert tensor to numpy for computation
        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth labels, one row per sample
        Returns:
            a tensor of floats, one per sample
        '''

        dev = x.device
        x.to_host()
        y.to_host()

        x_np = tensor.to_numpy(x)
        y_np = tensor.to_numpy(y)

        print 'x_np shape: ', x_np.shape
        print 'y_np shape: ', y_np.shape

        pred_x_np = x_np > 0.0

        accuracy = tensor.from_numpy(pred_x_np == y_np) # still a matrix
        macro_auc = roc_auc_score(y_np, x_np, average='macro')
        micro_auc = roc_auc_score(y_np, x_np, average='micro')


        x.to_device(dev)
        y.to_device(dev)
        accuracy.to_device(dev)

        return [accuracy, macro_auc, micro_auc]

    def evaluate(self, x, y):
        '''Compute the averaged precision over all samples.
        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth values, one row per sample
        Returns:
            a float value for the averaged metric
        '''
        accuracy, macro_auc, micro_auc = self.forward(x, y)
        return [tensor.average(accuracy), macro_auc, micro_auc]
