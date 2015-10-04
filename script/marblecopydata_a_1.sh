#!/bin/bash
hostfile="/home/singa/zhaojing/script/marbleslaves_a_1"

hosts=(`cat $hostfile |cut -d ' ' -f1`)

#for i in 1 2 3; do ssh awan-0-0$i-0 "hostname"; done

#for i in ${hosts[@]}
for i in `cat /home/singa/zhaojing/script/marbleslaves_a_1`
do
    echo $i
      echo "ssh $i"
        # ssh $i "mkdir /data/zhaojing/"
        # ssh $i "scp -r singa@logbase-a05:/data/zhaojing/SynPUF-regularization/ /data/zhaojing/"
        # ssh $i "scp -r singa@logbase-a05:/data/zhaojing/CMSHF-regularization/ /data/zhaojing/"
        # ssh $i  "scp -r singa@logbase-a05:/data/zhaojing/SynPUF-regularization/SynPUF_2009_refer_2010_Car_Cla_Vec_Regulariz_test_shard/ /data/zhaojing/SynPUF-regularization/"
        # ssh $i "scp -r singa@logbase-gw:/data/zhaojing/software/ /data/zhaojing/"
        ssh $i "scp -r singa@logbase-gw:/data/zhaojing/marble/ /data/zhaojing/"
      done
