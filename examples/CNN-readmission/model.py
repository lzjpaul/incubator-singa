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
from singa import layer
from singa import metric
from singa import loss
from singa import initializer
# from singa import net as ffnet
import mynet as myffnet


def create_net(in_shape, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'

    net = myffnet.ProbFeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D('conv1', 100, kernel=(3, 80), stride=(1, 20), pad=(0, 0),
                         input_sample_shape=in_shape))
    net.add(layer.Activation('relu1'))
    net.add(layer.MaxPooling2D('pool1', 2, 1, pad=0))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('dense', 2))

    for (pname, pvalue) in zip(net.param_names(), net.param_values()):
        if len(pvalue.shape) > 1:
            initializer.gaussian(pvalue, pvalue.shape[0], pvalue.shape[1])
        else:
            pvalue.set_value(0)
        print pname, pvalue.l1()
    return net