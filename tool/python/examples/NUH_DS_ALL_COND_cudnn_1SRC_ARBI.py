#!/usr/bin/env python
# import os and create folder line 11-19
# create model files
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from singa.model import *
from examples.datasets import NUHALLCOND1SRCARBI
import numpy as np
import random
from random import randint
import numpy
import os

X_train, X_test, X_valid, workspace = NUHALLCOND1SRCARBI.load_data()

version_num = random.randint(0,10000)
data_dir_prefix = '/data/zhaojing/result/1-31-CNN-1SRC-ARBI'
workspace = data_dir_prefix + '/version' + str(version_num)
if not os.path.exists(workspace):
    os.mkdir(workspace)

Uniform_or_Gaussian = int (sys.argv[1])
b_Uniform_or_Constant = random.randint(0,1)
input_y = int (sys.argv[2]) # calculate inner size
input_x = int (sys.argv[3]) # calculate inner size

kernel_x_param_array = np.array([60, 66, 80, 140, 300])
kernel_y_param_array = np.array([2, 3])
stride_x_param_array = np.array([20, 25, 30,35])
stride_y_param_array = 1
kernel_x_param = kernel_x_param_array[random.randint(0,len(kernel_x_param_array)-1)]
kernel_y_param = kernel_y_param_array[random.randint(0,len(kernel_y_param_array)-1)]
#stride_x_param = stride_x_param_array[random.randint(0,len(stride_x_param_array)-1)]
if kernel_x_param < 80:
    stride_x_param = stride_x_param_array[random.randint(0,0)]
elif kernel_x_param == 80:
    stride_x_param = stride_x_param_array[random.randint(0,1)]
else:
    stride_x_param = stride_x_param_array[random.randint(0,len(stride_x_param_array)-1)]
# stride_y_param = stride_y_param_array[random.randint(0,len(stride_y_param_array))]
stride_y_param = stride_y_param_array
input_channel = 1
pad_x_param = 2
pad_y_param = 0
filter_num_param = 500

pool_param_array = np.array([2,3])
pool_x_param = pool_y_param = pool_param_array[random.randint(0,len(pool_param_array)-1)]
pool_stride_param_array = np.array([pool_x_param, pool_x_param-1])
if pool_x_param == 2:
    pool_stride_x_param = pool_stride_y_param = pool_stride_param_array[random.randint(0,len(pool_stride_param_array)-1)]
else:
    pool_stride_x_param = pool_stride_y_param = 1

pool_pad_x_param = pool_pad_y_param = 0

conv_fan_in = input_channel * kernel_y_param * kernel_x_param
conv_fan_out = filter_num_param * kernel_y_param * kernel_x_param 
conv_fan_out_1 = (filter_num_param * kernel_y_param * kernel_x_param) / (pool_x_param * pool_y_param)

conv_n_array = np.array([conv_fan_in, (conv_fan_in + conv_fan_out)/2, conv_fan_out, (conv_fan_in + conv_fan_out_1)/2, conv_fan_out_1])

#gaussian
conv_std_w_param = numpy.sqrt(2. / (conv_n_array[random.randint(0,len(conv_n_array)-1)]))
conv_constant_b_param = 0
#uniform, b can use 0 also
conv_uniform_w_param = numpy.sqrt(3. / (conv_n_array[random.randint(0,len(conv_n_array)-1)]))
conv_uniform_b_param = conv_uniform_w_param

feamap_y = (input_y - kernel_y_param + 2*pad_y_param) / stride_y_param + 1
feamap_x = (input_x - kernel_x_param + 2*pad_x_param) / stride_x_param + 1
pool_map_y = (feamap_y - pool_y_param + 2*pool_pad_y_param) / pool_stride_y_param + 1
pool_map_x = (feamap_x - pool_x_param + 2*pool_pad_x_param) / pool_stride_x_param + 1

softmax_fan_in = filter_num_param * pool_map_y * pool_map_x
softmax_fan_out = 2
softmax_n_array = np.array([softmax_fan_in,(softmax_fan_in + softmax_fan_out)/2]) # do not use fan_out

#gaussian
softmax_std_w_param = numpy.sqrt(2. / (softmax_n_array[random.randint(0,len(softmax_n_array)-1)]))
softmax_constant_b_param = 0
#uniform, b can use 0 also
softmax_uniform_w_param = numpy.sqrt(3. / (softmax_n_array[random.randint(0,len(softmax_n_array)-1)]))
softmax_uniform_b_param = softmax_uniform_w_param


lr_array = np.array([0.1, 0.01, 0.001, 0.0001])
decay_array = np.array([0.01, 0.001, 0.0001])
momentum_array = np.array([0.8, 0.9])
lr_param = lr_array[random.randint(0,len(lr_array)-1)]
decay_param = decay_array[random.randint(0,len(decay_array)-1)]
momentum_param = momentum_array[random.randint(0,len(momentum_array)-1)]

m = Sequential('NUHALLCOND-cnn', sys.argv)
#gaussian
conv_parw_gaussian = Parameter(init='gaussian', std=conv_std_w_param)
conv_parb_gaussian = Parameter(init='constant', value=conv_constant_b_param)
#uniform
conv_parw_uniform = Parameter(init='uniform', scale=conv_uniform_w_param)
conv_parb_uniform = Parameter(init='uniform', scale=conv_uniform_b_param)
conv_parb_uniform_1 = Parameter(init='constant', value=conv_constant_b_param)

if Uniform_or_Gaussian == 1:
    m.add(Convolution2D(filter_num_param, pad_x=pad_x_param, pad_y=pad_y_param, stride_x=stride_x_param, stride_y=stride_y_param, kernel_x=kernel_x_param, kernel_y=kernel_y_param, w_param=conv_parw_gaussian, b_param=conv_parb_gaussian))
elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 0:
    m.add(Convolution2D(filter_num_param, pad_x=pad_x_param, pad_y=pad_y_param, stride_x=stride_x_param, stride_y=stride_y_param, kernel_x=kernel_x_param, kernel_y=kernel_y_param, w_param=conv_parw_uniform, b_param=conv_parb_uniform))
elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 1:
    m.add(Convolution2D(filter_num_param, pad_x=pad_x_param, pad_y=pad_y_param, stride_x=stride_x_param, stride_y=stride_y_param, kernel_x=kernel_x_param, kernel_y=kernel_y_param, w_param=conv_parw_uniform, b_param=conv_parb_uniform_1))

m.add(Activation('relu'))
m.add(MaxPooling2D(pool_size=(pool_x_param,pool_y_param), stride=pool_stride_x_param))

#gaussian
softmax_parw_gaussian = Parameter(init='gaussian', std=softmax_std_w_param)
softmax_parb_gaussian = Parameter(init='constant', value=softmax_constant_b_param)
#uniform
softmax_parw_uniform = Parameter(init='uniform', scale=softmax_uniform_w_param)
softmax_parb_uniform = Parameter(init='uniform', scale=softmax_uniform_b_param)
softmax_parb_uniform_1 = Parameter(init='constant', value=softmax_constant_b_param)

if Uniform_or_Gaussian == 1: 
    m.add(Dense(2, w_param=softmax_parw_gaussian, b_param=softmax_parb_gaussian, activation='softmax'))
elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 0:
    m.add(Dense(2, w_param=softmax_parw_uniform, b_param=softmax_parb_uniform, activation='softmax'))
elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 1:
    m.add(Dense(2, w_param=softmax_parw_uniform, b_param=softmax_parb_uniform_1, activation='softmax'))

# sgd = SGD(decay=0.004, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001),momentum=0.9, lr=0.001)
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
# m.fit(X_train, nb_epoch=7000, with_test=True, device=gpu_id)
result = m.evaluate(X_test, test_steps=30, test_freq=20)
# ./bin/singa-run.sh -exec tool/python/examples/NUH_DS_ALL_COND_cudnn_tune_init.py 0 12 1277
