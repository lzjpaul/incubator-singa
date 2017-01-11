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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import metric
from singa import loss
from singa import net as ffnet

# check pipeline
# intialization correct?
def create_net(in_shape, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'

    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.AUC())
    #W0_specs = {'init': 'gaussian', 'mean': 0, 'std': 0.0001}
    #W1_specs = {'init': 'gaussian', 'mean': 0, 'std': 0.01}
    #W2_specs = {'init': 'gaussian', 'mean': 0, 'std': 0.01, 'decay_mult': 250}
    # b_specs = {'init': 'constant', 'value': 0, 'lr_mult': 2, 'decay_mult': 0}

    net.add(layer.Conv2D('conv1', 100, kernel=(3, 80), stride=(1, 20), pad=(0, 2),
                         input_sample_shape=in_shape))
    net.add(layer.Activation('relu1'))
    net.add(layer.MaxPooling2D('pool1', 2, 1, pad=0))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('dense', 2))
    #for (p, specs) in zip(net.param_values(), net.param_specs()):
    #    filler = specs.filler
    #    if filler.type == 'gaussian':
    #        p.gaussian(filler.mean, filler.std)
    #    else:
    #        p.set_value(0)
    #    print specs.name, filler.type, p.l1()
    for (pname, pvalue) in zip(net.param_names(), net.param_values()):
        if len(pvalue.shape) > 1:
            initializer.gaussian(pvalue, pvalue.shape[0], pvalue.shape[1])
        else:
            pvalue.set_value(0)
        print pname, pvalue.l1()
    return net
