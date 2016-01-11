#!/usr/bin/env python
from singa.model import *

def load_data(
         workspace = None, 
         backend = 'kvfile',
         shape = (1, 12, 1277)
      ):

  # using cifar10 dataset
  data_dir = '/data/zhaojing/cnn/NUHALLCOND/subsample1'
  path_train = data_dir + '/train_data_normrange_ARBI.bin'
  path_test  = data_dir + '/test_data_normrange_ARBI.bin'
  path_valid = data_dir + '/valid_data_normrange_ARBI.bin'
  if workspace == None: workspace = data_dir

  store = Store(path=path_train, backend=backend,
              batchsize=100,
              shape=shape) 

  data_train = Data(load='recordinput', phase='train', conf=store)

  store = Store(path=path_test, backend=backend,
              batchsize=100,
              shape=shape) 

  data_test = Data(load='recordinput', phase='test', conf=store)

  store = Store(path=path_valid, backend=backend,
              batchsize=100,
              shape=shape) 

  data_valid = Data(load='recordinput', phase='val', conf=store)

  return data_train, data_test, data_valid, workspace

