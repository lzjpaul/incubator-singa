ssh logbase-a12 nohup python /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/marbleTests_zj_argu.py 25 /data/zhaojing/marble/tensor/subs1.csv /data/zhaojing/marble/tensor/vals1.csv /data/zhaojing/marble/tensor/siz.csv 3 /data/zhaojing/marble/tensor/subs2.csv /data/zhaojing/marble/tensor/vals2.csv /data/zhaojing/marble/tensor/siz.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test.txt > /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/redirecte2-1 2>&1 &
echo "ssh a12"
ssh logbase-a13 nohup python /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/marbleTests_zj_argu.py 25 /data/zhaojing/marble/tensor/subs1.csv /data/zhaojing/marble/tensor/vals1.csv /data/zhaojing/marble/tensor/siz.csv 3 /data/zhaojing/marble/tensor/subs2.csv /data/zhaojing/marble/tensor/vals2.csv /data/zhaojing/marble/tensor/siz.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_1.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_1.txt > /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/redirecte2-2 2>&1 &
echo "ssh a13"
ssh logbase-a14 nohup python /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/marbleTests_zj_argu.py 25 /data/zhaojing/marble/tensor/subs1.csv /data/zhaojing/marble/tensor/vals1.csv /data/zhaojing/marble/tensor/siz.csv 3 /data/zhaojing/marble/tensor/subs2.csv /data/zhaojing/marble/tensor/vals2.csv /data/zhaojing/marble/tensor/siz.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_2.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_2.txt > /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/redirecte2-3 2>&1 &
echo "ssh a14"
