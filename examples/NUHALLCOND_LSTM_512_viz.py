'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.utils.visualize_util import plot
import random
import sys


batch_size = 100 #modify for test and valid
nb_classes = 2
nb_epoch = 20

print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=False, input_shape=(12, 1253)))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()
plot(model, to_file='model.png')

# sudo python NUHALLCOND_LSTM.py /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv
