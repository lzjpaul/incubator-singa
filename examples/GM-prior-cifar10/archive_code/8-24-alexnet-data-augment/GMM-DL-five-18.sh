python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 2 -hyperparampath script_generator/8-24/hyperparams/lambda_1_a_1_alpha_0_b_0.csv -resultpath /data/zhaojing/regularization/log0824/GMM-DL-five/GMM-DL-five_lambda_1_a_1_alpha_0_b_0_result.txt alexnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0824/GMM-DL-five/GMM-DL-five-18.log