import numpy as np

w_array = np.array([1,2,3])
w_array = np.reshape(w_array, (1,3))
w_array = np.reshape(w_array, w_array.shape[1])
    # grad = param * w.sign()
    # print "lasso w shape: ", w.shape
theta_vec = np.array([0.3, 0.4, 0.5])
lambda_vec = np.array([1/4., 1/9., 1/16.])
grad_denominator = np.zeros(w_array.shape[0])
grad_numerator = np.zeros(w_array.shape[0])
for i in range(theta_vec.shape[0]):
    grad_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array)
    print "grad_denominator_inc: ", grad_denominator_inc
    grad_denominator = grad_denominator + grad_denominator_inc
    print "grad_denominator: ", grad_denominator
for i in range(theta_vec.shape[0]):
    grad_numerator_inc = theta_vec[i] * np.power((lambda_vec[i]/ (2 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array) * lambda_vec[i] * w_array
    print "grad_numerator_inc: ", grad_numerator_inc
    grad_numerator = grad_numerator + grad_numerator_inc
    print "grad_numerator: ", grad_numerator
grad = grad_numerator / grad_denominator
print "grad: ", grad

#for i in range(theta_vec.shape[0]):
#        grad_denominator = grad_denominator + theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array)
#for i in range(theta_vec.shape[0]):
#        grad_numerator = grad_numerator + theta_vec[i] * np.power((lambda_vec[i]/ (2 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array * w_array) * lambda_vec[i] * w_array
