import numpy as np

theta_vec = np.array([0.1, 0.2, 0.3])
lambda_vec = np.array([1., 2., 3.])
w_array = np.array([1., 2., 3., 4.])
for i in range(theta_vec.shape[0]):
    # print "gaussian theta i: ", i
    res_denominator_inc = theta_vec[i] * np.power((lambda_vec[i] / (2.0 * np.pi)), 0.5) * np.exp(-0.5 * lambda_vec[i] * w_array[:-1] * w_array[:-1])
    print res_denominator_inc
