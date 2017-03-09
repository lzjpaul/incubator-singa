import numpy as np

theta_vec = np.array([0.1, 0.2, 0.3])
theta_alpha = 2
print "theta_alpha: ", theta_alpha
res_matrix = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9],[1.0, 1.1, 1.2]])
print "res_matrix shape: ", res_matrix.shape
print "res_matrix: ", res_matrix
###theta_minimizer#########
theta_numerator = np.sum(res_matrix, axis=0) + theta_alpha - 1
print "theta_numerator: ", theta_numerator
theta_denominator = np.sum(res_matrix) + theta_vec.shape[0] * (theta_alpha - 1)
print "theta_denominator: ", theta_denominator
theta_vec_minimizer = (theta_numerator / theta_denominator)
print "theta_vec_minimizer: ", theta_vec_minimizer
print "sum theta_vec_minimizer: ", np.sum(theta_vec_minimizer)
