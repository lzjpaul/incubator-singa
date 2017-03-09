from scipy.stats import norm as gaussian
import numpy as np
theta_vec = np.array([0.2, 0.3, 0.5])
lambda_vec = np.array([0.1, 0.2, 0.3])
w_array = np.array([1,2,3,4,5,6,7,8,9,10])
res_denominator = np.zeros(9)
for i in range(theta_vec.shape[0]):
    # print "gaussian theta i: ", i
    res_denominator_inc = theta_vec[i] * lambda_vec[i] * np.exp(lambda_vec[i] + w_array[:-1] * w_array[:-1])
    print "res_denominator_inc: ", res_denominator_inc
    print "res_denominator_inc shape: ", res_denominator_inc.shape
    if i == 0:
        res_matrix = np.reshape(res_denominator_inc, (-1, 1))
    else:
        res_matrix = np.concatenate((res_matrix, np.reshape(res_denominator_inc, (-1, 1))), axis=1)
    res_denominator = res_denominator + res_denominator_inc
    print "res_denominator: ", res_denominator
    print "res_matrix: ", res_matrix
res_matrix = res_matrix / res_denominator.reshape((-1,1)).astype(float)
print "res_matrix: ", res_matrix
            # print "res_matrix shape: ", res_matrix.shape
        ##update responsibility##
print "np.sum(res_matrix, axis=1): ", np.sum(res_matrix, axis=1)

