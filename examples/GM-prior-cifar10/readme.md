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
(1) parameter gaussian initialization 
(2) mini-batch / queue are fixed order

