python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 -hyperparampath script_generator/9-8-resnet-new-scale/hyperparams/lambda_1_a_1_alpha_1_b_1.csv -resultpath /data/zhaojing/regularization/log0908/GMM-DL-seven-resnet-new-scale/GMM-DL-seven-resnet-new-scale_lambda_1_a_1_alpha_1_b_1_result.txt resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0908/GMM-DL-seven-resnet-new-scale/GMM-DL-seven-resnet-new-scale-27.log
