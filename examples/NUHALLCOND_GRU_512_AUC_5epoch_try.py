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
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file
from sklearn.metrics import roc_auc_score
import random
import sys

y_score = np.zeros(10)
test_label_matrix = np.zeros(10)
c = numpy.asarray(y_score, dtype = float)
numpy.savetxt("GRU_512_5_epoch_prediction.txt", c, fmt = '%6f', delimiter=",") #modify here
d = numpy.asarray(test_label_matrix, dtype = int)
numpy.savetxt("GRU_512_5_epoch_label.txt", d, fmt = '%d', delimiter=",") #modify here
# sudo python NUHALLCOND_GRU.py /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_train_label_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_test_label_normrange_1.csv
