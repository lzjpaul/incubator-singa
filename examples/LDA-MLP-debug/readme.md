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

## begining tune parameters (away from 0.9204 for resnet)

2017-9-11
(1) mini-batch are using less memory operations (only augment in batch manner)

2017-9-12
(1) linear initialization: base * float(gm_num)

2017-9-13
(1) turn back to large memory mini-batch

2017-12-22
(1) set weight decay to zero
    (1) all use L2-norm
    (2) all weight decay to zero
    (3) no AUC currently
(2) no printings
(3) using float32 to read in label
(4) momentum: 0.0 (ada gradient)
(5) adding sigmoid layer!!
(6) in AUCAcfuray loss layer, need to manually sigmoid!!

2017-12-24
(1) lda_register: only one lda regularizer is registered!
(2) in phi, for some words, all topic is zero!!
(3) for log(phi*theta), some are zero in log!!
