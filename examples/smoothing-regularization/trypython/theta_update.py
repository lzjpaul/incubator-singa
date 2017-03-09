import numpy as np

res_matrix = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9],[1.0, 1.1, 1.2]])
theta_vec = np.array([0.1, 0.2, 0.3])
theta_alpha = 2
theta_r_vec = np.zeros(3)
###theta_r_update#########
print "theta_vec: ", theta_vec
print "res_matrix: ", res_matrix
theta_k_r_j = np.zeros((theta_vec.shape[0], theta_vec.shape[0])) ## derivative of theta_k to r_j
for k in range(theta_k_r_j[:,0].shape[0]):
    for j in range(theta_k_r_j[0, :].shape[0]):
        if k == j:
            theta_k_r_j[k, j] = (theta_vec[k] - theta_vec[k] * theta_vec[k])
        else:
              theta_k_r_j[k, j] = (- theta_vec[k] * theta_vec[j])
print "theta_k_r_j: ", theta_k_r_j
theta_r_vec_update = np.zeros(theta_r_vec.shape[0])
term1 = (theta_alpha - 1) / theta_vec.astype(float)
print "term1: ", term1
term2 = np.sum(res_matrix.astype(float) / (theta_vec.reshape(1, -1)), axis=0)
print "res_matrix.astype(float) / (theta_vec.reshape(1, -1)): ", res_matrix.astype(float) / (theta_vec.reshape(1, -1))
print "term2: ", term2
theta_derivative = ( - term1 - term2)
print "theta_derivative: ", theta_derivative
for j in range(theta_r_vec_update.shape[0]): #r_j
    print "theta_k_r_j[:, j]: ", theta_k_r_j[:, j]
    theta_r_vec_update[j] = np.sum(theta_derivative * theta_k_r_j[:, j])

print "theta_r_vec_update: ", theta_r_vec_update

responsibility = res_matrix
pi = theta_vec.reshape((1,-1))
alpha = theta_alpha
gm_num = 3
delta_pi = np.sum(responsibility / pi.astype(float), axis=0) + (alpha - 1) / pi.astype(float)
delta_pi = -delta_pi
delta_pi_k_j_mat = np.array([[(int(j==k)*pi[0,j] - pi[0,j] *pi[0,k]) for j in range(gm_num)] for k in range(gm_num)])
print "delta_pi_k_j_mat: ", delta_pi_k_j_mat
delta_pi_r = np.matmul(delta_pi, delta_pi_k_j_mat)
print "delta_pi_r: ", delta_pi_r
