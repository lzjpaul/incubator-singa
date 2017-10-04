python train_gpu_0.py resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-baseline-0.log
python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 -hyperparampath script_generator/9-12-resnet-alpha-init/hyperparams/lambda_0_a_0_alpha_0_b_0.csv -resultpath /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1_lambda_0_a_0_alpha_0_b_0_result.txt resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-0.log
python gm_prior_train.py -maxepoch 10 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 -hyperparampath script_generator/9-12-resnet-alpha-init/hyperparams/lambda_0_a_0_alpha_1_b_0.csv -resultpath /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1_lambda_0_a_0_alpha_1_b_0_result.txt resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0912/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1/GMM-DL-tune-three-resnet-alpha-init-olm-last-try-1-1.log