python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 1 -hyperparampath script_generator/8-24/hyperparams/lambda_1_a_0_alpha_1_b_1.csv -resultpath /data/zhaojing/regularization/log0824/GMM-DL-five/GMM-DL-five_lambda_1_a_0_alpha_1_b_1_result.txt alexnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0824/GMM-DL-five/GMM-DL-five-15.log