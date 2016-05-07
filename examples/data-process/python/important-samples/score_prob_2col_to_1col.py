# two columns: (sample num, label)
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read training data
file = open(sys.argv[1])
two_col_label_matrix = np.genfromtxt(file, delimiter=",")
file.close()
print "2col_label_matrix.shape =\n", two_col_label_matrix.shape

two_col_label_matrix = two_col_label_matrix.astype(np.int)
#print "correct_0_index = \n", 2col_label_matrix

#read training data
file = open(sys.argv[2])
two_col_prediction_matrix = np.genfromtxt(file, delimiter=",")
file.close()
print "2col_prediction_matrix.shape =\n", two_col_prediction_matrix.shape

two_col_prediction_matrix = two_col_prediction_matrix.astype(np.float)
#print "correct_0_index = \n", 2col_label_matrix

one_col_label_matrix = two_col_label_matrix[:,1]
one_col_prediction_matrix = two_col_prediction_matrix[:,1]

#for i in range(len(one_col_prediction_matrix)):
#	if one_col_label_matrix[i] == 0:
#	     one_col_prediction_matrix[i] = two_col_prediction_matrix[i,0]

#output
a = numpy.asarray(one_col_label_matrix, dtype = int)
numpy.savetxt(sys.argv[3], a, fmt = '%d', delimiter=",") #modify here
a = numpy.asarray(one_col_prediction_matrix, dtype = float)
numpy.savetxt(sys.argv[4], a, fmt = '%6f', delimiter=",") #modify here
# python score_prob_2col_to_1col.py LSTM_1024_5_epoch/LSTM_1024_5_epoch_label.txt LSTM_1024_5_epoch/LSTM_1024_5_epoch_prediction.txt LSTM_1024_5_epoch/LSTM_1024_5_epoch_label_1col.csv LSTM_1024_5_epoch/LSTM_1024_5_epoch_prediction_1col.csv
