# add AUC after run
# nb_epoch is 2,4,6,8,10 customized
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
from sklearn.metrics import roc_auc_score
import random
import sys


batch_size = 100 #modify for test and valid
nb_classes = 2
nb_epoch = 20

#read data
file = open(sys.argv[1])
train_data_matrix = np.genfromtxt(file, delimiter=",")
train_data_matrix = train_data_matrix.astype('float32')
file.close()
print ("train_data_matrix = ", train_data_matrix.shape)

#read data
file = open(sys.argv[2])
train_label_matrix = np.genfromtxt(file, delimiter=",")
train_label_matrix = train_label_matrix.astype(np.int)
file.close()
print ("train_label_matrix = ", train_label_matrix.shape)

#read data
file = open(sys.argv[3])
test_data_matrix = np.genfromtxt(file, delimiter=",")
test_data_matrix = test_data_matrix.astype('float32')
file.close()
print ("test_data_matrix = ", test_data_matrix.shape)

#read data
file = open(sys.argv[4])
test_label_matrix = np.genfromtxt(file, delimiter=",")
test_label_matrix = test_label_matrix.astype(np.int)
file.close()
print ("test_label_matrix = ", test_label_matrix.shape)

# the data, shuffled and split between train and test sets
case_num = 12
case_feature_dim = 1253

train_data_matrix = train_data_matrix.reshape(4850, case_num, case_feature_dim)
test_data_matrix = test_data_matrix.reshape(2910, case_num, case_feature_dim)
print ("train_data_matrix = ", train_data_matrix.shape)
print ("test_data_matrix = ", test_data_matrix.shape)

nb_classes = 2
train_label_matrix = np_utils.to_categorical(train_label_matrix, nb_classes)
test_label_matrix = np_utils.to_categorical(test_label_matrix, nb_classes)
# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Build model...')
model = Sequential()
model.add(LSTM(2048, return_sequences=False, input_shape=(case_num, case_feature_dim)))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_data_matrix, train_label_matrix,
                    batch_size=batch_size, nb_epoch=11,
                    verbose=1, validation_data=(test_data_matrix, test_label_matrix))
score = model.evaluate(test_data_matrix, test_label_matrix, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
y_score = model.predict_proba(test_data_matrix)
print ("roc")
print ("roc = ", roc_auc_score(test_label_matrix, y_score))

# sudo python NUHALLCOND_LSTM.py /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv
