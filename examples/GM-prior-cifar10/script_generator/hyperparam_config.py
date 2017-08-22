import numpy as np
import os

if __name__ == '__main__':
    b_list, alpha_list = [100., 10., 1., 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001], [0.7, 0.5, 0.3]
    a_list = [1e-1, 1e-2]
    gm_lambda_ratio_list = [ -1., 0.05,  1.]
    
    # modify here
    lambda_idx_list = [np.array([0,1]), np.array([1,2]), np.array([2,3])]
    a_idx_list=[np.array([0,1]), np.array([1,2])]
    alpha_idx_list = [np.array([0,1]), np.array([1,2]), np.array([2,3])]
    b_idx_list = [np.array([0,5]), np.array([5, 9])]
    
    for i in range(len(lambda_idx_list)):
        for j in range(len(a_idx_list)):
            for k in range(len(alpha_idx_list)):
                for l in range(len(b_idx_list)):
                    hyperparam_config = np.zeros((4,2)) # lambda, a, alpha, b
                    hyperparam_config = hyperparam_config.astype(int)
                    hyperparam_config[0] = lambda_idx_list[i]
                    hyperparam_config[1] = a_idx_list[j]
                    hyperparam_config[2] = alpha_idx_list[k]
                    hyperparam_config[3] = b_idx_list[l]
                    hyperparam_file_name = 'lambda_' + str(i) + '_a_' + str(j) + '_alpha_' + str(k) + '_b_' + str(l) + '.csv'
                    hyperparam_path = '8-22/hyperparams'
                    print os.path.join(hyperparam_path, hyperparam_file_name)
                    np.savetxt(os.path.join(hyperparam_path, hyperparam_file_name), hyperparam_config, fmt = '%d', delimiter=",")
                    # test
                    print hyperparam_config
                    gm_lambda_ratio_list_new = gm_lambda_ratio_list[hyperparam_config[0][0]:hyperparam_config[0][1]]
                    a_list_new = a_list[hyperparam_config[1][0]:hyperparam_config[1][1]]
                    alpha_list_new = alpha_list[hyperparam_config[2][0]:hyperparam_config[2][1]]
                    b_list_new = b_list[hyperparam_config[3][0]:hyperparam_config[3][1]]
                    print "gm_lambda_ratio_list_new", gm_lambda_ratio_list_new
                    print "a_list_new", a_list_new
                    print "alpha_list_new", alpha_list_new
                    print "b_list_new", b_list_new
