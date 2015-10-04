#!/bin/bash
hostfile="/home/singa/zhaojing/script/cpydataslaves"

hosts=(`cat $hostfile |cut -d ' ' -f1`)

#for i in 1 2 3; do ssh awan-0-0$i-0 "hostname"; done

#for i in ${hosts[@]}
for i in `cat /home/singa/zhaojing/script/cpydataslaves`
do
    echo $i
      echo "ssh $i"
        # ssh $i "mkdir /data/zhaojing/"
        # ssh $i "scp -r singa@logbase-a05:/data/zhaojing/SynPUF-regularization/ /data/zhaojing/"
        # ssh $i "scp -r singa@logbase-a05:/data/zhaojing/CMSHF-regularization/ /data/zhaojing/"
         ssh $i  "scp -r singa@logbase-a05:/data/zhaojing/SynPUF-regularization/SynPUF_2009_refer_2010_Car_Cla_Vec_Regulariz_test_shard/ /data/zhaojing/SynPUF-regularization/"
      done
