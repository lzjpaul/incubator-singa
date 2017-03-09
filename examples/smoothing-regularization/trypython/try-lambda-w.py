import numpy as np
w_array = np.array([1,2,3,4,5])
lambda_vec = np.array([1, 2, 3])
res_matrix = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9],[1.0, 1.1, 1.2]])
w_weight_array = w_array[:-1].reshape((-1, 1)) # no bias
# grad = param * w.sign()
# print "lasso w shape: ", w.shape
print "bias: ", w_array[-1]
lambda_w = np.dot(w_weight_array, lambda_vec.reshape((1, -1)))
print "lambda_w: ", lambda_w
print "res_matrix: ", res_matrix
grad = np.zeros(w_array.shape[0])
grad[:-1] = np.sum((res_matrix * lambda_w), axis=1)# -log(p(w))
print "res_matrix * lambda_w: ", res_matrix * lambda_w
print "grad: ", grad

w = w_weight_array.reshape((-1, 1))
reg_lambda = lambda_vec.reshape((1, -1))
responsibility = res_matrix
reg_grad_w = np.sum(responsibility*reg_lambda, axis=1).reshape(w.shape) * w
print "reg_grad_w: ", reg_grad_w
print "np.vstack((reg_grad_w, np.array([0.0]))): ", np.vstack((reg_grad_w, np.array([0.0])))
