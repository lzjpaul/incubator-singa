name: "CMS-cnn-pearson"
neuralnet {
layer {
  name: "data0"
  include: kTrain
  type: kRecordInput
  partition_dim: 0
  store_conf {
    backend: "kvfile"
    path: "/ssd/zhaojing/cnn/CMS/subsample2/normrange_DIAGPROC_PEARSON/train_data.bin"
    batchsize: 100
    shape: 1
    shape: 24
    shape: 338
  }
}
layer {
  name: "data0"
  include: kTest
  type: kRecordInput
  partition_dim: 0
  store_conf {
    backend: "kvfile"
    path: "/ssd/zhaojing/cnn/CMS/subsample2/normrange_DIAGPROC_PEARSON/test_data.bin"
    batchsize: 100
    shape: 1
    shape: 24
    shape: 338
  }
}
layer {
  name: "data0"
  include: kVal
  type: kRecordInput
  partition_dim: 0
  store_conf {
    backend: "kvfile"
    path: "/ssd/zhaojing/cnn/CMS/subsample2/normrange_DIAGPROC_PEARSON/valid_data.bin"
    batchsize: 100
    shape: 1
    shape: 24
    shape: 338
  }
}
layer {
  name: "conv1"
  srclayers: "data0"
  param {
    name: "w1"
    init {
      type: kUniform
      low: -0.12403473
      high: 0.12403473
    }
    lr_scale: 1
    wd_scale: 1
  }
  param {
    name: "b1"
    init {
      type: kConstant
      value: 0
    }
    lr_scale: 1
    wd_scale: 1
  }
  type: kCudnnConv
  partition_dim: 0
  convolution_conf {
    num_filters: 750
    kernel_x: 65
    kernel_y: 3
    pad_x: 0
    pad_y: 0
    stride_x: 10
    stride_y: 1
  }
}
layer {
  name: "relu1"
  srclayers: "conv1"
  type: kCudnnActivation
  partition_dim: 0
  activation_conf {
    type: RELU
  }
}
layer {
  name: "pool1"
  srclayers: "relu1"
  type: kCudnnPool
  partition_dim: 0
  pooling_conf {
    kernel: 2
    pool: MAX
    pad: 0
    stride: 2
  }
}
layer {
  name: "layer2"
  srclayers: "pool1"
  param {
    name: "w2"
    init {
      type: kUniform
      low: -0.0072074374
      high: 0.0072074374
    }
    lr_scale: 1
    wd_scale: 1
  }
  param {
    name: "b2"
    init {
      type: kConstant
      value: 0
    }
    lr_scale: 1
    wd_scale: 1
  }
  type: kInnerProduct
  partition_dim: 0
  innerproduct_conf {
    num_output: 2
  }
}
layer {
  name: "softmax2"
  srclayers: "layer2"
  srclayers: "data0"
  type: kCudnnSoftmaxLoss
  partition_dim: 0
  softmaxloss_conf {
    topk: 1
  }
}
}
train_one_batch {
  alg: kBP
}
updater {
  type: kAdaGrad
  learning_rate{
    base_lr: 0.01
    type: kFixed
  }
  momentum: 0.8
  weight_decay: 0.01
}
cluster {
  nworker_groups: 1
  nserver_groups: 1
  nworkers_per_group: 1
  nservers_per_group: 1
  nworkers_per_procs: 1
  nservers_per_procs: 1
  workspace: "/ssd/zhaojing/cnn/CMS/result/CNN-PEARSON/subsample2"
}
train_steps: 12000
disp_freq: 1200
gpu: 1
test_freq: 20
test_steps: 30
validate_freq: 20
validate_steps: 20
checkpoint_freq: 12000
