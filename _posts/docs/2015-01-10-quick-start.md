---
layout: post
title: Quick Start
category : docs
tags : [installation, examples]
---
{% include JB/setup %}


## Installation

Clone the SINGA code from [Github](https://github.com/apache/incubator-singa)
or Apache's git repository

    git clone git@github.com:apache/incubator-singa.git
    or
    git clone https://github.com/apache/incubator-singa.git


Compile SINGA:

    ./configure
    make

If there are dependent libraries missing, please refer to
[installation]({{ BASE_PATH }}{% post_url /docs/2015-01-20-installation %}) page
for guidance on installing them. After successful compilation, the libsinga.so
and singa executable will be built into the build folder.

## Run in standalone mode

Running SINGA in standalone mode is on the contrary of running it on Mesos or
YARN. For standalone mode, users have to manage the resources manually. For
instance, they have to prepare a host file containing all running nodes.
There is no management on CPU and memory resources, hence SINGA consumes as much
CPU and memory resources as it needs.

### Training on a single node

For single node training, one process will be launched to run the SINGA code on
the node where SINGA is started. We train the [CNN model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) over the
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset as an example.
The hyper-parameters are set following
[cuda-convnet](https://code.google.com/p/cuda-convnet/).


#### Data and model preparation

Download the dataset and create the data shards for training and testing.

    cd examples/cifar10/
    make download
    make create

A training dataset and a test dataset are created under *train-shard* and
*test-shard* folder respectively. A image_mean.bin file is also generated, which
contains the feature mean of all images.
<!--After creating the data shards, you  to update the paths in the
model configuration file (*model.conf*) for the
training data shard, test data shard and the mean file.-->

Since all modules used for training this CNN model are provided by SINGA as
built-in modules, there is no need to write any code. Instead, you just run the
executable file (*../../build/singa*) by providing the model configuration file
(*model.conf*).  If you want to implement your own modules, e.g., layer,
then you have to register your modules in the driver code. After compiling the
driver code, link it with the SINGA library to generate the executable. More
details are described in [Code your own models]().

#### Training without partitioning

To train the model without any partitioning, you just set the numbers
in the cluster configuration file (*cluster.conf*) as :

    nworker_groups: 1
    nworkers_per_group: 1
    nserver_groups: 1
    nservers_per_group: 1

One worker group trains against one partition of the training dataset. If
*nworker_groups* is set to 1, then there is no data partitioning. One worker
runs over a partition of the model. If *nworkers_per_group* is set to 1, then
there is no model partitioning. More details on the cluster configuration are
described in the [System Architecture]() page.

Start the training by running:

    #goto top level folder
    cd ..
    ./singa -model=examples/cifar10/model.conf -cluster=examples/cifar10/cluster.conf

#### Training with data Partitioning

#### Training with model Partitioning

### Training in a cluster


## Run with Mesos

*in working*...

## Run with YARN
