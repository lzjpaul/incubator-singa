python gm_prior_train_no_data_augment.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 -hyperparampath script_generator/9-10-alexnet-new-scale/hyperparams/lambda_0_a_0_alpha_3_b_0.csv -resultpath /data/zhaojing/regularization/log0910/GMM-DL-eleven-alexnet-new-scale/GMM-DL-eleven-alexnet-new-scale_lambda_0_a_0_alpha_3_b_0_result.txt alexnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0910/GMM-DL-eleven-alexnet-new-scale/GMM-DL-eleven-alexnet-new-scale-3.log
