python train.py alexnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0812/GMM-DL-one/GMM-DL-one-1
python train.py resnet cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0812/GMM-DL-one/GMM-DL-one-1
python train.py vgg cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0812/GMM-DL-one/GMM-DL-one-1
python gm_prior_train.py -alexnetdim 89440 -vggdim 14977728 -resnetdim 270896 -maxepoch 100 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 vgg cifar-10-batches-py/ | tee -a /data/zhaojing/regularization/log0812/GMM-DL-one/GMM-DL-one-1