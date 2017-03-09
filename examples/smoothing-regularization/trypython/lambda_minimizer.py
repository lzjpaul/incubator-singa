import numpy as np
from scipy import sparse
a = 2
b = 2
res_matrix = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9],[1.0, 1.1, 1.2]])
w_weight_array = np.array([1, 2, 3, 4]).reshape((4,1))
print "res_matrix: ", res_matrix
###lambda_minimizer#########
lambda_numerator = 0.5 * res_matrix
print "lambda_numerator: ", lambda_numerator
lambda_numerator = np.sum(lambda_numerator, axis=0)
print "lambda_numerator: ", lambda_numerator
lambda_numerator = lambda_numerator + a -1
print "lambda_numerator: ", lambda_numerator
# print "w_weight_array shape: ", w_weight_array.shape
# print "diag shape: ", np.diag(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])).shape
# print "res_matrix.T shape: ", res_matrix.T.shape
lambda_denominator = sparse.csr_matrix(res_matrix.T).dot(sparse.diags(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])))
lambda_denominator = (lambda_denominator.toarray().T)
print "lambda_denominator: ", lambda_denominator
lambda_denominator = np.sum(lambda_denominator, axis=0)
lambda_denominator = lambda_denominator + b
print "lambda_denominator: ", lambda_denominator
lambda_vec_minimizer = (lambda_numerator / lambda_denominator.astype(float))
print "lambda_vec_minimizer: ", lambda_vec_minimizer
###lambda_minimizer#########

