python gm_prior_train_no_data_augment.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 1 -hyperparampath script_generator/8-25/hyperparams/lambda_2_a_0_alpha_0_b_0.csv -resultpath /data/zhaojing/regularization/log0825/GMM-DL-six/GMM-DL-six_lambda_2_a_0_alpha_0_b_0_result.txt alexnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0825/GMM-DL-six/GMM-DL-six-24.log
