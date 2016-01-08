#!/usr/bin/env python
# initialization method
# Sequential(...)
# valid dataset?
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from singa.model import *
from examples.datasets import NUHALLCOND

X_train, X_test, X_valid, workspace = NUHALLCOND.load_data()
# X_train, X_test, workspace = NUHALLCOND.load_data()

m = Sequential('NUHALLCOND-cnn', sys.argv)

# m.add(Convolution2D(500,  init='uniform', scale=0.1, pad_y=0, pad_x=10, stride_x=35, stride_y=1, kernel_x=140, kernel_y=3, b_lr=2.0, b_wd=0, w_std=0.0001))
# m.add(Convolution2D(500,  init='uniform', scale=0.1, pad_y=0, pad_x=10, stride_x=35, stride_y=1, kernel_x=140, kernel_y=3))
m.add(Convolution2D(500,  init='uniform', scale=0.1, pad_x=2, pad_y=0, stride_x=20, stride_y=1, kernel_x=60, kernel_y=3))
m.add(Activation('relu'))
m.add(MaxPooling2D(pool_size=(3,3), stride=2))

m.add(Dense(2, init='uniform', scale=0.1, activation='softmax'))

sgd = SGD(decay=0.004, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001),momentum=0.9, lr=0.001)
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)

gpu_id = [1]
m.fit(X_train, nb_epoch=1400, with_test=True, validate_data=X_valid, validate_steps=20, validate_freq=20, device=gpu_id)
# m.fit(X_train, nb_epoch=7000, with_test=True, device=gpu_id)
result = m.evaluate(X_test, test_steps=30, test_freq=20)
