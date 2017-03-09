import numpy as np
from scipy.stats import norm as gaussian

w_array = np.array([1,2,3])
w_array = np.reshape(w_array, (1,3))
w_array = np.reshape(w_array, w_array.shape[1])
    # grad = param * w.sign()
    # print "lasso w shape: ", w.shape
theta_vec = np.array([0.3, 0.4, 0.2, 0.3])
lambda_vec = np.array([1/4., 1/9., 1/16., 1/25.])
grad_denominator = np.zeros(w_array.shape[0])
grad_numerator = np.zeros(w_array.shape[0])
for i in range(theta_vec.shape[0]):
    grad_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array)
    print "grad_denominator_inc: ", grad_denominator_inc
    grad_denominator = grad_denominator + grad_denominator_inc
    print "grad_denominator: ", grad_denominator
for i in range(theta_vec.shape[0]):
    grad_numerator_inc = theta_vec[i] * np.power((lambda_vec[i]/ (2 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array)
    # print "grad_numerator_inc: ", grad_numerator_inc
    grad_numerator = grad_numerator + grad_numerator_inc
    print "grad_numerator_inc/grad_denominator: ", grad_numerator_inc / grad_denominator.astype(float)

res_denominator = np.zeros(3)
for i in range(theta_vec.shape[0]):
    res_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array)
    if i == 0:
        res_matrix = np.reshape(res_denominator_inc, (-1, 1))
    else:
        res_matrix = np.concatenate((res_matrix, np.reshape(res_denominator_inc, (-1, 1))), axis=1)
    res_denominator = res_denominator + res_denominator_inc
res_matrix = res_matrix / res_denominator.reshape((-1,1)).astype(float)
##update responsibility##
print "res_matrix: ", res_matrix

pi = theta_vec.reshape((1,4))
reg_lambda = lambda_vec.reshape((1,4))
w = w_array.reshape((3,1))

responsibility = gaussian.pdf(w, loc=np.zeros(shape=(1, 4)), scale=1/np.sqrt(reg_lambda))*pi
responsibility = responsibility/(np.sum(responsibility, axis=1).reshape(w.shape))
print "responsibility: ", responsibility
#for i in range(theta_vec.shape[0]):
#        grad_denominator = grad_denominator + theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array)
#for i in range(theta_vec.shape[0]):
#        grad_numerator = grad_numerator + theta_vec[i] * np.power((lambda_vec[i]/ (2 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array) * lambda_vec[i] * w_array
