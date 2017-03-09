import numpy as np
from scipy import sparse
w_array = np.array([1,2,3,4,5])
lambda_vec = np.array([1, 2, 3])
res_matrix = np.array([[0.1, 0.2, 0.9],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9],[1.0, 1.1, 1.2]])
w_weight_array = w_array[:-1].reshape((-1, 1)) # no bias
a = 20
b = 10
###lambda_t_update#########
term1 = (float(a-1) / lambda_vec) - b
print "term1: ", term1
print "res_matrix: ", res_matrix
res_w = sparse.csr_matrix(res_matrix.T).dot(sparse.diags(0.5 * w_weight_array.reshape(w_weight_array.shape[0]) * w_weight_array.reshape(w_weight_array.shape[0])))
res_w = (res_w.toarray().T)
print "res_w: ", res_w
term2 = np.sum((res_matrix / (2.0 * lambda_vec.reshape(1, -1))) - res_w, axis=0)
print "(res_matrix / (2.0 * lambda_vec.reshape(1, -1))): ", (res_matrix / (2.0 * lambda_vec.reshape(1, -1)))
print "(res_matrix / (2.0 * lambda_vec.reshape(1, -1))) - res_w: ", (res_matrix / (2.0 * lambda_vec.reshape(1, -1))) - res_w
print "term2: ", term2
print "(- term1 - term2): ", (- term1 - term2)
lambda_t_vec_update = (- term1 - term2) * lambda_vec #derivative of lambda to t
print "lambda_t_vec_update: ", lambda_t_vec_update

reg_lambda = lambda_vec.reshape((1,-1))
w = w_array.reshape((-1,1))
responsibility = res_matrix
delta_reg_lambda = np.sum((responsibility / (2.0 * reg_lambda) - responsibility * 0.5 * w[:-1] * w[:-1]), axis=0).reshape((1,-1))
delta_reg_lambda += (a - 1) / (reg_lambda.astype(float)) - b
delta_reg_lambda = -delta_reg_lambda
delta_reg_lambda_s = delta_reg_lambda * reg_lambda
print "delta_reg_lambda_s: ", delta_reg_lambda_s
