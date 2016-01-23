#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import * 
from examples.datasets import NUHALLCOND2SRCPEARSON_3
import random
from random import randint
import numpy
import numpy as np
import os

# Sample parameter values for Mnist MLP example
# pvalues = {'batchsize' : 100, 'shape' : 15324, 'random_skip' : 0}
X_train, X_test, X_valid, workspace = NUHALLCOND2SRCPEARSON_3.load_data()

version_num = random.randint(0,10000)
data_dir_prefix = '/data/zhaojing/result/1-23-MLP-Nlayer-PEARSON-3'
workspace = data_dir_prefix + '/version' + str(version_num)
if not os.path.exists(workspace):
    os.mkdir(workspace)

b_Uniform_or_Constant = random.randint(0,1)
m = Sequential('mlp', argv=sys.argv)

''' Weight and Bias are initialized by
    uniform distribution with scale=0.05 at default
'''
layer_neuron_size_array = np.array([100, 500, 1000, 2000, 4000, 8000])
layer_num = random.randint(1,3)

mlp_fan_out = 0
mlp_fan_in = 20532
for i in range(layer_num):
    if i == 0:
        mlp_fan_out = layer_neuron_size = layer_neuron_size_array[random.randint(0,len(layer_neuron_size_array)-2)] # no 8000 for the 1st layer
    else:
        mlp_fan_out = layer_neuron_size = layer_neuron_size_array[random.randint(0,len(layer_neuron_size_array)-1)]
    mlp_n_array = np.array([mlp_fan_in,(mlp_fan_in + mlp_fan_out)/2, mlp_fan_out])
    mlp_uniform_w_param = numpy.sqrt(3. / (mlp_n_array[random.randint(0,len(mlp_n_array)-1)]))
    mlp_uniform_b_param = mlp_uniform_w_param
    mlp_parw_uniform = Parameter(init='uniform', scale=mlp_uniform_w_param)
    mlp_parb_uniform = Parameter(init='uniform', scale=mlp_uniform_b_param)
    mlp_parb_uniform_1 = Parameter(init='constant', value=0)
    if b_Uniform_or_Constant == 0:
        m.add(Dense(layer_neuron_size, w_param=mlp_parw_uniform, b_param=mlp_parb_uniform, activation='tanh'))
    else:
        m.add(Dense(layer_neuron_size, w_param=mlp_parw_uniform, b_param=mlp_parb_uniform_1, activation='tanh'))
    mlp_fan_in = mlp_fan_out

softmax_fan_in = mlp_fan_out
softmax_n_array = np.array([softmax_fan_in,(softmax_fan_in + 2)/2])
softmax_uniform_w_param = numpy.sqrt(3. / (softmax_n_array[random.randint(0,len(softmax_n_array)-1)]))
softmax_uniform_b_param = softmax_uniform_w_param
softmax_parw_uniform = Parameter(init='uniform', scale=softmax_uniform_w_param)
softmax_parb_uniform = Parameter(init='uniform', scale=softmax_uniform_b_param)
softmax_parb_uniform_1 = Parameter(init='constant', value=0)
if b_Uniform_or_Constant == 0:
    m.add(Dense(2, w_param=softmax_parw_uniform, b_param=softmax_parb_uniform, activation='softmax'))
else:
    m.add(Dense(2, w_param=softmax_parw_uniform, b_param=softmax_parb_uniform_1, activation='softmax'))

lr_array = np.array([0.1, 0.01, 0.001, 0.0001])
decay_array = np.array([0.1, 0.01, 0.001, 0.0001])
momentum_array = np.array([0.8, 0.9])
lr_param = lr_array[random.randint(0,len(lr_array)-1)]
decay_param = decay_array[random.randint(0,len(decay_array)-1)]
momentum_param = momentum_array[random.randint(0,len(momentum_array)-1)]

ada = AdaGrad(lr=lr_param, decay=decay_param, momentum=momentum_param)
f = open(workspace + '/model', 'w+')
f.write("lr_param: " + str(lr_param))
f.write("\n")
f.write("decay_param: " + str(decay_param))
f.write("\n")
f.write("momentum_param: " + str(momentum_param))
f.write("\n")
f.close()

topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=ada, cluster=topo)

gpu_id = [0]
m.fit(X_train, nb_epoch=12000, with_test=True, validate_data=X_valid, validate_steps=20, validate_freq=20, device=gpu_id)
result = m.evaluate(X_test, test_steps=30, test_freq=20)
#e.g., display result
#for k, v in sorted(result.items(), key=lambda x: x[0]):
#  print k, v
