./bin/singa-run.sh -model=examples/marble/regularized/mlp2level-01-decay-001-level4-sub1.conf -cluster=examples/marble/cluster.conf 2>&1 | tee /home/singa/zhaojing/NUH-singa/incubator-singa/examples/marble/regularized/redirect/p01-w001-l4-s1-levelcomp
./bin/singa-run.sh -model=examples/marble/regularized/mlp2level-01-decay-001-level4-sub2.conf -cluster=examples/marble/cluster.conf 2>&1 | tee /home/singa/zhaojing/NUH-singa/incubator-singa/examples/marble/regularized/redirect/p01-w001-l4-s2-levelcomp
./bin/singa-run.sh -model=examples/marble/regularized/mlp2level-01-decay-001-level4-sub3.conf -cluster=examples/marble/cluster.conf 2>&1 | tee /home/singa/zhaojing/NUH-singa/incubator-singa/examples/marble/regularized/redirect/p01-w001-l4-s3-levelcomp
./bin/singa-run.sh -model=examples/marble/regularized/mlp2level-01-decay-001-level4-sub4.conf -cluster=examples/marble/cluster.conf 2>&1 | tee /home/singa/zhaojing/NUH-singa/incubator-singa/examples/marble/regularized/redirect/p01-w001-l4-s4-levelcomp
