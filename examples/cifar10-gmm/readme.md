(1) code
train-debug-1(modify a_val and b_val) --> copy to train.py

train-wd-0004-norm.py (come from train-debug-1, but fixed 0004, remember to opt.register(p), tensor.l1()&l2() norm)

(2) modifications
a_val and b_val (train-debug-1, and then copy train-debug-1 to update train.py)
alexnet.py(modify weight_decay and lr_rate)
opt.register(p)!!! (train-wd-0004-norm.py)
modify train-debug-1.py according to train-wd-0004-norm.py

(3) others:
/home/zhaojing/try-compile/incubator-singa/examples/cifar10
having the compiled codeee!!!

(4) code and places:
/home/zhaojing/try-wheel/pysinga + /home/zhaojing/smoothing-regularization/incubator-singa/examples/cifar10-gmm/train-wd-0004-norm.py
/home/zhaojing/try-compile/incubator-singa/pysinga +  /home/zhaojing/try-compile/incubator-singa/examples/cifar10/train-origin-wd-0004-norm.py

(5) different from alexnet(SINGA):
5-1: bias: lr_multipiler = 2
5-2: dense_weight:  decay_multiplier = 250
