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

# convert class vectors to binary class matrices
train_label_matrix = np_utils.to_categorical(train_label_matrix, nb_classes)
test_label_matrix = np_utils.to_categorical(test_label_matrix, nb_classes)

layer_num = random.randint(1,5)
hidden_unit_num_array = np.array([100, 200, 500, 1000, 2000, 3500])
model = Sequential()
model.add(Dense(hidden_unit_num_array[random.randint(0,len(hidden_unit_num_array)-1)], input_shape=(1253,)))
model.add(Activation('tanh'))
# model.add(Dropout(0.2))
if layer_num > 1:
    for i in range(layer_num-1):
        model.add(Dense(hidden_unit_num_array[random.randint(0,len(hidden_unit_num_array)-1)]))
        model.add(Activation('tanh'))
# model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
nb_epoch_ = 0
if layer_num == 1:
   nb_epoch_ = random.randint(2,4)
elif layer_num == 2:
   nb_epoch_ = random.randint(3,6)
elif layer_num == 3:
   nb_epoch_ = random.randint(4,8)
elif layer_num == 4:
   nb_epoch_ = random.randint(4,10)
elif layer_num == 5:
   nb_epoch_ = random.randint(4,10)

history = model.fit(train_data_matrix, train_label_matrix,
                    batch_size=batch_size, nb_epoch=nb_epoch_,
                    verbose=1, validation_data=(test_data_matrix, test_label_matrix))
score = model.evaluate(test_data_matrix, test_label_matrix, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
y_score = model.predict_proba(test_data_matrix)
print ("roc")
print ("roc = ", roc_auc_score(test_label_matrix, y_score))
f = open("MLP_NUHALLCOND_1_visit.csv", 'a')
f.write(str(roc_auc_score(test_label_matrix, y_score)) + "," + str(score[1]) + "\n")
f.close()
