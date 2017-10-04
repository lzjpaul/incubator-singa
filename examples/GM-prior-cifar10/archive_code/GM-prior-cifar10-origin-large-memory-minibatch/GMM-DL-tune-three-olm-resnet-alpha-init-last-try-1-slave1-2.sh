python train_gpu_2.py resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-baseline-2.log
python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 2 -hyperparampath script_generator/9-12-resnet-alpha-init/hyperparams/lambda_1_a_1_alpha_0_b_0.csv -resultpath /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1_lambda_1_a_1_alpha_0_b_0_result.txt resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-12.log
python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 2 -hyperparampath script_generator/9-12-resnet-alpha-init/hyperparams/lambda_1_a_1_alpha_1_b_0.csv -resultpath /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1_lambda_1_a_1_alpha_1_b_0_result.txt resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-13.log
python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 2 -hyperparampath script_generator/9-12-resnet-alpha-init/hyperparams/lambda_1_a_1_alpha_2_b_0.csv -resultpath /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1_lambda_1_a_1_alpha_2_b_0_result.txt resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-14.log
python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 2 -hyperparampath script_generator/9-12-resnet-alpha-init/hyperparams/lambda_1_a_1_alpha_3_b_0.csv -resultpath /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1_lambda_1_a_1_alpha_3_b_0_result.txt resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-15.log