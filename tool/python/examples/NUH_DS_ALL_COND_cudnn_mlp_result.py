#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import * 
from examples.datasets import NUHALLCONDLONGVEC
from examples.datasets import NUHALLCONDLONGVECSHAPE
import random
from random import randint
import numpy
import numpy as np
import os

# Sample parameter values for Mnist MLP example
pvalues = {'batchsize' : 100, 'shape' : 15324, 'random_skip' : 0}
X_train, X_test, X_valid, workspace = NUHALLCONDLONGVECSHAPE.load_data()

version_num = random.randint(0,10000)
data_dir_prefix = '/data/zhaojing/result/1-16-mlp'
workspace = data_dir_prefix + '/version' + str(version_num)
if not os.path.exists(workspace):
    os.mkdir(workspace)

m = Sequential('mlp', argv=sys.argv)

''' Weight and Bias are initialized by
    uniform distribution with scale=0.05 at default
'''
layer_neuron_size_array = np.array([100, 500, 1000, 2000, 4000, 8000])
layer_num = random.randint(1,1)

fan_in = 15324
for i in range(layer_num):
    if i == 0:
        fan_out = layer_neuron_size = layer_neuron_size_array[random.randint(0,len(layer_neuron_size_array)-2)] # no 8000 for the 1st layer
    else:
        fan_out = layer_neuron_size = layer_neuron_size_array[random.randint(0,len(layer_neuron_size_array)-1)]
    m.add(Dense(layer_neuron_size, init='uniform', scale=numpy.sqrt(6. / (fan_in + fan_out)), activation='tanh'))
    fan_in = fan_out

m.add(Dense(2, init='uniform', scale=numpy.sqrt(6. / (fan_in + 2)), activation='softmax')) 

lr_array = np.array([0.1, 0.01, 0.001, 0.0001])
decay_array = np.array([0.01, 0.001, 0.0001])
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

#gpu_id = [0]
m.fit(X_train, nb_epoch=12000, with_test=True, validate_data=X_valid, validate_steps=20, validate_freq=20)
result = m.evaluate(X_test, test_steps=30, test_freq=20)
#e.g., display result
#for k, v in sorted(result.items(), key=lambda x: x[0]):
#  print k, v
