#!/usr/bin/env python
# should moify diffrent layer's stride (first layer can not be too small!!) and kernel size
# PEARSON and KB: PEARASON should have 300, KB should not, otherwise can not explain!! 
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from singa.model import *
from examples.datasets import NUHALLCOND2SRCPEARSON_3
import numpy as np
import random
from random import randint
import numpy
import os

X_train, X_test, X_valid, workspace = NUHALLCOND2SRCPEARSON_3.load_data()

version_num = random.randint(0,10000)
data_dir_prefix = '/data/zhaojing/result/1-23-CNN-Nlayer-PEARSON-3'
workspace = data_dir_prefix + '/version' + str(version_num)
if not os.path.exists(workspace):
    os.mkdir(workspace)

Uniform_or_Gaussian = int (sys.argv[1])
b_Uniform_or_Constant = random.randint(0,1)
Add_MLP_or_not = random.randint(0,1)
layer_num = random.randint(1,3)
input_y = int (sys.argv[2]) # calculate inner size
input_x = int (sys.argv[3]) # calculate inner size
#convolution layers

input_channel = 1
filter_num_param_array = np.array([150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800])
kernel_x_param_array = np.array([3, 3, 5, 10, 15, 20, 60, 66, 80, 140, 300])
kernel_y_param_array = np.array([2, 3])
stride_x_param_array = np.array([1, 2, 3, 20, 25, 30, 35])
stride_y_param_array = np.array([1, 2, 3])

kernel_x_param = np.zeros(layer_num)
kernel_y_param = np.zeros(layer_num)
stride_x_param = np.zeros(layer_num)
stride_y_param = np.zeros(layer_num)
pad_x_param = np.zeros(layer_num)
pad_y_param = np.zeros(layer_num)
filter_num_param = np.zeros(layer_num)


for i in range(layer_num):
    if i == 0:
        kernel_x_param[i] = kernel_x_param_array[random.randint(6,len(kernel_x_param_array)-1)]
        kernel_y_param[i] = kernel_y_param_array[random.randint(0,len(kernel_y_param_array)-1)]
        if kernel_x_param[i] <= 80:
            stride_x_param[i] = stride_x_param_array[random.randint(3,3)]
        else:
            stride_x_param[i] = stride_x_param_array[random.randint(3,len(stride_x_param_array)-1)]
        stride_y_param[i] = stride_y_param_array[random.randint(0,len(stride_y_param_array)-2)] # 1,2
        filter_num_param[i] = filter_num_param_array[random.randint(5,len(filter_num_param_array)-2)] # 250, 300, 400, 500
    elif i == 1:
        kernel_x_param[i] = kernel_x_param_array[random.randint(0,5)]
        kernel_y_param[i] = kernel_y_param_array[random.randint(0,len(kernel_y_param_array)-1)]
        if kernel_x_param[i] <= 5:
            stride_x_param[i] = stride_x_param_array[random.randint(0,0)] #1
        else:
            stride_x_param[i] = stride_x_param_array[random.randint(0,2)] #1,2,3
        stride_y_param[i] = stride_y_param_array[random.randint(0,len(stride_y_param_array)-2)] # 1,2
        filter_num_param[i] = filter_num_param_array[random.randint(5,len(filter_num_param_array)-2)] # 400, 450, 500, ..., 700
    else:
        kernel_x_param[i] = kernel_x_param_array[random.randint(0,2)]
        kernel_y_param[i] = kernel_y_param_array[random.randint(0,len(kernel_y_param_array)-1)]
        if kernel_x_param[i] <= 5:
            stride_x_param[i] = stride_x_param_array[random.randint(0,0)] #1
        else:
            stride_x_param[i] = stride_x_param_array[random.randint(0,2)] #1,2,3
        stride_y_param[i] = stride_y_param_array[random.randint(0,len(stride_y_param_array)-2)]
        filter_num_param[i] = filter_num_param_array[random.randint(7,len(filter_num_param_array)-1)] # 500, 600, ..., 800

#pooling layers
pool_param_array = np.array([2,3])
pool_x_param = pool_y_param = pool_param_array[random.randint(0,len(pool_param_array)-1)]
pool_stride_param_array = np.array([pool_x_param-1, pool_x_param])
if pool_x_param == 3:
    pool_stride_x_param = pool_stride_y_param = pool_stride_param_array[random.randint(0,len(pool_stride_param_array)-2)]
else:
    pool_stride_x_param = pool_stride_y_param = 1
pool_pad_x_param = pool_pad_y_param = 0


#calculate fan_in and fan_out
conv_fan_in = np.zeros(layer_num)
conv_fan_out = np.zeros(layer_num)
conv_fan_out_1 = np.zeros(layer_num)

for i in range(layer_num):
    if i == 0:
        conv_fan_in[i] = input_channel * kernel_y_param[i] * kernel_x_param[i]
    else:
        conv_fan_in[i] = filter_num_param[i-1] * kernel_y_param[i] * kernel_x_param[i]

    conv_fan_out[i] = filter_num_param[i] * kernel_y_param[i] * kernel_x_param[i]

    if i == (layer_num - 1):
        conv_fan_out_1[i] = (filter_num_param[i] * kernel_y_param[i] * kernel_x_param[i]) / (pool_x_param * pool_y_param)
    else:
        conv_fan_out_1[i] = (filter_num_param[i] * kernel_y_param[i] * kernel_x_param[i]) / (kernel_y_param[i+1] * kernel_x_param[i+1])


# calculate feamap, in order for last layer fully-connected
feamap_y = np.zeros(layer_num)
feamap_x = np.zeros(layer_num)

for i in range(layer_num):
    if i == 0:
        feamap_y[i] = (input_y - kernel_y_param[i] + 2*pad_y_param[i]) / stride_y_param[i] + 1
        feamap_x[i] = (input_x - kernel_x_param[i] + 2*pad_x_param[i]) / stride_x_param[i] + 1
    else:
        feamap_y[i] = (feamap_y[i-1] - kernel_y_param[i] + 2*pad_y_param[i]) / stride_y_param[i] + 1
        feamap_x[i] = (feamap_x[i-1] - kernel_x_param[i] + 2*pad_x_param[i]) / stride_x_param[i] + 1

pool_map_y = (feamap_y[layer_num-1] - pool_y_param + 2*pool_pad_y_param) / pool_stride_y_param + 1
pool_map_x = (feamap_x[layer_num-1] - pool_x_param + 2*pool_pad_x_param) / pool_stride_x_param + 1


lr_array = np.array([0.1, 0.01, 0.001, 0.0001])
decay_array = np.array([0.1, 0.01, 0.001, 0.0001])
momentum_array = np.array([0.8, 0.9])
lr_param = lr_array[random.randint(0,len(lr_array)-1)]
decay_param = decay_array[random.randint(0,len(decay_array)-1)]
momentum_param = momentum_array[random.randint(0,len(momentum_array)-1)]

m = Sequential('NUHALLCOND-cnn', sys.argv)

# convert to int
kernel_x_param = kernel_x_param.astype(int)
kernel_y_param = kernel_y_param.astype(int)
stride_x_param = stride_x_param.astype(int)
stride_y_param = stride_y_param.astype(int)
pad_x_param = pad_x_param.astype(int)
pad_y_param = pad_y_param.astype(int)
filter_num_param = filter_num_param.astype(int)

# convolution and relu
for i in range(layer_num):
    conv_n_array = np.array([conv_fan_in[i], (conv_fan_in[i] + conv_fan_out[i])/2, conv_fan_out[i], (conv_fan_in[i] + conv_fan_out_1[i])/2, conv_fan_out_1[i]])
    #gaussian
    conv_std_w_param = numpy.sqrt(2. / (conv_n_array[random.randint(0,len(conv_n_array)-1)]))
    conv_constant_b_param = 0
    #uniform, b can use 0 also
    conv_uniform_w_param = numpy.sqrt(3. / (conv_n_array[random.randint(0,len(conv_n_array)-1)]))
    conv_uniform_b_param = conv_uniform_w_param
    #gaussian
    conv_parw_gaussian = Parameter(init='gaussian', std=conv_std_w_param)
    conv_parb_gaussian = Parameter(init='constant', value=conv_constant_b_param)
    #uniform
    conv_parw_uniform = Parameter(init='uniform', scale=conv_uniform_w_param)
    conv_parb_uniform = Parameter(init='uniform', scale=conv_uniform_b_param)
    conv_parb_uniform_1 = Parameter(init='constant', value=conv_constant_b_param)

    if Uniform_or_Gaussian == 1:
        m.add(Convolution2D(filter_num_param[i], pad_x=pad_x_param[i], pad_y=pad_y_param[i], stride_x=stride_x_param[i], stride_y=stride_y_param[i], kernel_x=kernel_x_param[i], kernel_y=kernel_y_param[i], w_param=conv_parw_gaussian, b_param=conv_parb_gaussian))
    elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 0:
        m.add(Convolution2D(filter_num_param[i], pad_x=pad_x_param[i], pad_y=pad_y_param[i], stride_x=stride_x_param[i], stride_y=stride_y_param[i], kernel_x=kernel_x_param[i], kernel_y=kernel_y_param[i], w_param=conv_parw_uniform, b_param=conv_parb_uniform))
    elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 1:
        m.add(Convolution2D(filter_num_param[i], pad_x=pad_x_param[i], pad_y=pad_y_param[i], stride_x=stride_x_param[i], stride_y=stride_y_param[i], kernel_x=kernel_x_param[i], kernel_y=kernel_y_param[i], w_param=conv_parw_uniform, b_param=conv_parb_uniform_1))
    m.add(Activation('relu'))


m.add(MaxPooling2D(pool_size=(pool_x_param,pool_y_param), stride=pool_stride_x_param))

if Add_MLP_or_not == 1:
    layer_neuron_size_array = np.array([100, 300, 500, 750, 1000, 1500])
    mlp_fan_in = filter_num_param[layer_num-1] * pool_map_y * pool_map_x
    mlp_fan_out = layer_neuron_size = layer_neuron_size_array[random.randint(0,len(layer_neuron_size_array)-1)]
    mlp_n_array = np.array([mlp_fan_in,(mlp_fan_in + mlp_fan_out)/2, mlp_fan_out]) # do not use fan_out
    #gaussian
    mlp_std_w_param = numpy.sqrt(2. / (mlp_n_array[random.randint(0,len(mlp_n_array)-1)]))
    mlp_constant_b_param = 0
    #uniform, b can use 0 also
    mlp_uniform_w_param = numpy.sqrt(3. / (mlp_n_array[random.randint(0,len(mlp_n_array)-1)]))
    mlp_uniform_b_param = mlp_uniform_w_param
    #gaussian
    mlp_parw_gaussian = Parameter(init='gaussian', std=mlp_std_w_param)
    mlp_parb_gaussian = Parameter(init='constant', value=mlp_constant_b_param)
    #uniform
    mlp_parw_uniform = Parameter(init='uniform', scale=mlp_uniform_w_param)
    mlp_parb_uniform = Parameter(init='uniform', scale=mlp_uniform_b_param)
    mlp_parb_uniform_1 = Parameter(init='constant', value=mlp_constant_b_param)
    if Uniform_or_Gaussian == 1:
        m.add(Dense(layer_neuron_size, w_param=mlp_parw_gaussian, b_param=mlp_parb_gaussian, activation='relu'))
    elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 0:
        m.add(Dense(layer_neuron_size, w_param=mlp_parw_uniform, b_param=mlp_parb_uniform, activation='relu'))
    elif Uniform_or_Gaussian == 0 and b_Uniform_or_Constant == 1:
        m.add(Dense(layer_neuron_size, w_param=mlp_parw_uniform, b_param=mlp_parb_uniform_1, activation='relu'))
    softmax_fan_in = mlp_fan_out
    softmax_fan_out = 2
    softmax_n_array = np.array([softmax_fan_in,(softmax_fan_in + softmax_fan_out)/2]) # do not use fan_out

else:
    #softmax or add another fully connected
    softmax_fan_in = filter_num_param[layer_num-1] * pool_map_y * pool_map_x
    softmax_fan_out = 2
    softmax_n_array = np.array([softmax_fan_in,(softmax_fan_in + softmax_fan_out)/2]) # do not use fan_out

#gaussian
softmax_std_w_param = numpy.sqrt(2. / (softmax_n_array[random.randint(0,len(softmax_n_array)-1)]))
softmax_constant_b_param = 0
#uniform, b can use 0 also
softmax_uniform_w_param = numpy.sqrt(3. / (softmax_n_array[random.randint(0,len(softmax_n_array)-1)]))
softmax_uniform_b_param = softmax_uniform_w_param

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

gpu_id = [1]
m.fit(X_train, nb_epoch=12000, with_test=True, validate_data=X_valid, validate_steps=20, validate_freq=20, device=gpu_id)
result = m.evaluate(X_test, test_steps=30, test_freq=20)
# ./bin/singa-run.sh -exec tool/python/examples/NUH_DS_ALL_COND_cudnn_tune_init.py 0 12 1277
