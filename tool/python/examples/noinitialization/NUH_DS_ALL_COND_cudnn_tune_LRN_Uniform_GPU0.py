#!/usr/bin/env python
# uniform weight and bias have different initialization scale?
# Sequential(...)
# ada gradient? (steps?)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from singa.model import *
from examples.datasets import NUHALLCOND
import numpy as np
import random
from random import randint

X_train, X_test, X_valid, workspace = NUHALLCOND.load_data()
# X_train, X_test, workspace = NUHALLCOND.load_data()

kernel_x_param_array = np.array([60, 66, 80, 140, 200, 300])
kernel_y_param_array = np.array([2, 3])
stride_x_param_array = np.array([20, 25, 30, 35])
stride_y_param_array = 1
pool_param_array = np.array([2,3])
kernel_x_param = kernel_x_param_array[random.randint(0,len(kernel_x_param_array)-1)]
kernel_y_param = kernel_y_param_array[random.randint(0,len(kernel_y_param_array)-1)]
if kernel_x_param < 80:
    stride_x_param = stride_x_param_array[random.randint(0,0)]
elif kernel_x_param == 80:
    stride_x_param = stride_x_param_array[random.randint(0,1)]
else:
    stride_x_param = stride_x_param_array[random.randint(0,len(stride_x_param_array)-1)]
# stride_y_param = stride_y_param_array[random.randint(0,len(stride_y_param_array))]
stride_y_param = stride_y_param_array
pad_x_param = 2
pad_y_param = 0
pool_x_param = pool_y_param = pool_param_array[random.randint(0,len(pool_param_array)-1)]
pool_stride_param_array = np.array([pool_x_param, pool_x_param - 1])
pool_stride_param = pool_stride_param_array[random.randint(0,len(pool_stride_param_array)-1)]

LRN_local_size_array=np.array([3])
LRN_alpha_array=np.array([0.00005, 0.0005])
LRN_beta_array=np.array([0.75, 0.65])
LRN_local_size_param=LRN_local_size_array[random.randint(0,len(LRN_local_size_array)-1)]
LRN_alpha_param=LRN_alpha_array[random.randint(0,len(LRN_alpha_array)-1)]
LRN_beta_param=LRN_beta_array[random.randint(0,len(LRN_beta_array)-1)]


conv_w_scale_array = np.array([0.01, 0.05])
conv_b_scale_array = np.array([0.01, 0.05])
conv_scale_array = np.array([0.01, 0.05])
conv_w_scale_param = conv_w_scale_array[random.randint(0,len(conv_w_scale_array)-1)]
conv_b_scale_param = conv_b_scale_array[random.randint(0,len(conv_b_scale_array)-1)]
conv_scale_param = conv_scale_array[random.randint(0,len(conv_scale_array)-1)]

softmax_w_scale_array = np.array([0.01, 0.05])
softmax_b_scale_array = np.array([0.01, 0.05])
softmax_scale_array = np.array([0.01, 0.05])
softmax_w_scale_param = softmax_w_scale_array[random.randint(0,len(softmax_w_scale_array)-1)]
softmax_b_scale_param = softmax_b_scale_array[random.randint(0,len(softmax_b_scale_array)-1)]
softmax_scale_param = softmax_scale_array[random.randint(0,len(softmax_scale_array)-1)]
lr_array = np.array([0.0001, 0.001])
decay_array = np.array([0.0004, 0.004])
momentum_array = np.array([0.8, 0.9])
lr_param = lr_array[random.randint(0,len(lr_array)-1)]
decay_param = decay_array[random.randint(0,len(decay_array)-1)]
momentum_param = momentum_array[random.randint(0,len(momentum_array)-1)]

m = Sequential('NUHALLCOND-cnn', sys.argv)
conv_parw = Parameter(init='uniform', scale=conv_scale_param)
conv_parb = Parameter(init='uniform', scale=conv_scale_param)
m.add(Convolution2D(500, pad_x=pad_x_param, pad_y=pad_y_param, stride_x=stride_x_param, stride_y=stride_y_param, kernel_x=kernel_x_param, kernel_y=kernel_y_param, w_param=conv_parw, b_param=conv_parb))
m.add(Activation('relu'))
m.add(LRN2D(LRN_local_size_param, alpha=LRN_alpha_param, beta=LRN_beta_param))
m.add(MaxPooling2D(pool_size=(pool_x_param,pool_y_param), stride=pool_stride_param))

softmax_parw = Parameter(init='uniform', scale=softmax_scale_param)
softmax_parb = Parameter(init='uniform', scale=softmax_scale_param)
m.add(Dense(2, w_param=softmax_parw, b_param=softmax_parb, activation='softmax'))

# sgd = SGD(decay=0.004, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001),momentum=0.9, lr=0.001)
ada = AdaGrad(lr=lr_param, decay=decay_param, momentum=momentum_param)
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=ada, cluster=topo)

gpu_id = [0]
m.fit(X_train, nb_epoch=10000, with_test=True, validate_data=X_valid, validate_steps=20, validate_freq=20, device=gpu_id)
# m.fit(X_train, nb_epoch=7000, with_test=True, device=gpu_id)
result = m.evaluate(X_test, test_steps=30, test_freq=20)
