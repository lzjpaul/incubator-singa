import numpy as np
import os


class Config_Script:
    def __init__(self):
        self.gm_lambda_ratio_list = [ -1., 0.05,  1.]
        self.a_list = [1e-1, 1e-2]
        self.b_list, self.alpha_list = [100., 10., 1., 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001], [0.7, 0.5, 0.3]

        # modify here
        self.lambda_idx_list = [np.array([0,1]), np.array([1,2]), np.array([2,3])]
        self.a_idx_list=[np.array([0,1]), np.array([1,2])]
        self.alpha_idx_list = [np.array([0,1]), np.array([1,2]), np.array([2,3])]
        self.b_idx_list = [np.array([0,5]), np.array([5, 9])]

    def gen_hyperparam_config(self, hyperparam_path):
        index = 0
        for i in range(len(self.lambda_idx_list)):
            for j in range(len(self.a_idx_list)):
                for k in range(len(self.alpha_idx_list)):
                    for l in range(len(self.b_idx_list)):
                        hyperparam_config = np.zeros((4,2)) # lambda, a, alpha, b
                        hyperparam_config = hyperparam_config.astype(int)
                    	hyperparam_config[0] = self.lambda_idx_list[i]
                    	hyperparam_config[1] = self.a_idx_list[j]
                    	hyperparam_config[2] = self.alpha_idx_list[k]
                    	hyperparam_config[3] = self.b_idx_list[l]
                    	hyperparam_file_name = 'lambda_' + str(i) + '_a_' + str(j) + '_alpha_' + str(k) + '_b_' + str(l) + '.csv'
                    	# hyperparam_path = '8-22/hyperparams'
                    	hyperparam_path = hyperparam_path
                        print "\nindex: ", index
                        index = index + 1
                        print os.path.join(hyperparam_path, hyperparam_file_name)
                    	np.savetxt(os.path.join(hyperparam_path, hyperparam_file_name), hyperparam_config, fmt = '%d', delimiter=",")
                    	# test
                    	print hyperparam_config
                    	gm_lambda_ratio_list_new = self.gm_lambda_ratio_list[hyperparam_config[0][0]:hyperparam_config[0][1]]
                    	a_list_new = self.a_list[hyperparam_config[1][0]:hyperparam_config[1][1]]
                    	alpha_list_new = self.alpha_list[hyperparam_config[2][0]:hyperparam_config[2][1]]
                    	b_list_new = self.b_list[hyperparam_config[3][0]:hyperparam_config[3][1]]
                    	print "gm_lambda_ratio_list_new", gm_lambda_ratio_list_new
                    	print "a_list_new", a_list_new
                    	print "alpha_list_new", alpha_list_new
                    	print "b_list_new", b_list_new

    def gen_sh_script(self, sh_script_path):
        index = 0
        for i in range(len(self.lambda_idx_list)):
            for j in range(len(self.a_idx_list)):
                for k in range(len(self.alpha_idx_list)):
                    for l in range(len(self.b_idx_list)):
                        python_programme = "gm_prior_train.py "
                        hyper_param_path = "script_generator/8-23/hyperparams" # modify here
                        hyper_param_prefix = 'lambda_' + str(i) + '_a_' + str(j) + '_alpha_' + str(k) + '_b_' + str(l)
                        data_path = "/data/zhaojing/regularization/log0823/GMM-DL-four" # modify here
                        script = ("python " + python_programme + "-maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 -hyperparampath " + str(os.path.join(hyper_param_path, (hyper_param_prefix + ".csv"))) + " -resultpath " + str(os.path.join(data_path, ('GMM-DL-four_' + hyper_param_prefix + "_result "))) + "alexnet cifar-10-batches-py/ | tee -a " + data_path + "/GMM-DL-four-" + str(index) + ".log\n") # modify two places of 'GMM-DL-four-'
 
                        index = (index + 1)
                        f= open(sh_script_path, 'a')
                        f.write(script)
                        f.close()
                        # script = "python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 -hyperparampath 8-22/hyperparams/lambda_1_a_1_alpha_1_b_1.csv -resultpath /data/zhaojing/regularization/log0821/GMM-DL-three/lambda_1_a_1_alpha_1_b_1_result.csv resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0821/GMM-DL-three/GMM-DL-three-1"
        

    def gen_multiple_sh_scripts(self, sh_script_path, multiple_script_name):
        with open(sh_script_path, 'rb') as script_file:
            index = 0
            for line in script_file:
                f= open((multiple_script_name + str(index) + ".sh"), 'a')
                f.write(line)
                f.close()
                index = index + 1

if __name__ == '__main__':
   cs = Config_Script()
   cs.gen_hyperparam_config('8-23/hyperparams')
   cs.gen_sh_script('8-23/scripts/GMM-DL-four.sh') # need to go into to modify
   # need to modify gpu id manually first
   cs.gen_multiple_sh_scripts('8-23/scripts/GMM-DL-four.sh', '8-23/scripts/GMM-DL-four-')
