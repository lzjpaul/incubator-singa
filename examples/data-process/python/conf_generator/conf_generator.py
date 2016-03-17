#!/usr/bin/env python
# import os and create folder line 11-19
# create model files
import numpy as np
import random
from random import randint
import numpy
import os
import sys
import shutil

originfile = sys.argv[1]
lr_array = np.array([0.1, 0.01, 0.001, 0.0001])
decay_array = np.array([0.1, 0.01, 0.001, 0.0001])
for i in range(len(lr_array)):
    for j in range(len(decay_array)):
        shutil.copy(originfile,originfile+"lr"+str(i)+"decay"+str(j))
f = open(originfile+"lr"+str(0)+"decay"+str(0), 'a+b')
f.write("\nupdater {\n  type:kAdaGrad\n  learning_rate{\n    base_lr: " + str(lr_array[0]) + "\n    type: kFixed\n  }\n  momentum: 0.9\n  weight_decay: " + str(decay_array[0]) + "\n}")
f.write("\ncluster {\n  nworker_groups: 1\n  nserver_groups: 1\n  nworkers_per_group: 1\n  nservers_per_group: 1\n  nworkers_per_procs: 1\n  nservers_per_procs: 1\n  workspace: \"/ssd/zhaojing/cnn/NUHALLCOND/result/" + originfile+"lr"+str(i)+"decay"+str(j) + "\"")
print ("\ncluster {\n  nworker_groups: 1\n  nserver_groups: 1\n  nworkers_per_group: 1\n  nservers_per_group: 1\n  nworkers_per_procs: 1\n  nservers_per_procs: 1\n  workspace: \"/ssd/zhaojing/cnn/NUHALLCOND/result/" + originfile+"lr"+str(i)+"decay"+str(j) + "\"\n}" )
f.close()
