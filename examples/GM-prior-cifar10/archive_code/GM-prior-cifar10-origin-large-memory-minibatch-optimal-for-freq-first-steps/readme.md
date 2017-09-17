alexnet
89440

vgg
14977728

caffe
89440

resnet
270896

python gm_prior_train.py -alexnetdim 89440 -vggdim 14977728 -resnetdim 270896 -maxepoch 100 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 alexnet cifar-10-batches-py/

2017-8-20
tuning hyper-parameters:
(1) GM_num
(2) a, b, alpha
(3) pi, lambda

2017-9-7
away from 0.9206
(1) gaussian mixture initialization (using linear)
(2) mini-batch / queue are fixed order

2017-9-9
(1) the parameters scale are from 1,2,5,10 ...

2017-9-11
(1) mini-batch are using less memory operations (only augment in batch manner)

2017-9-11
(1) disable the get_hyperparams
gm_prior_train_optimal_for_freq.py
gm_prior_train_no_data_augment_optimal_for_freq.py

2017-9-12
(1) base * gm_num
(2) add one more parameter to the apply function to control first steps
(3) gm_prior_train_no_data_augment_optimal_for_freq_first_steps.py 

2017-9-15
(1) turn back to large memory mini-batch
