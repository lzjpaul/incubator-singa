ssh logbase-a04 nohup python /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/marbleTests_zj_argu.py 50 /data/zhaojing/marble/tensor/dataset1/subs_splittrain_1.csv /data/zhaojing/marble/tensor/dataset1/vals_binary_splittrain_1.csv /data/zhaojing/marble/tensor/siz.csv 150 /data/zhaojing/marble/tensor/dataset1/subs_splittest_1.csv /data/zhaojing/marble/tensor/dataset1/vals_binary_splittest_1.csv /data/zhaojing/marble/tensor/siz.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_S1_50_1.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_S1_50_1.txt > /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/redirect_S1_50_1 2>&1 &
echo "ssh logbase-a04"
ssh logbase-a05 nohup python /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/marbleTests_zj_argu.py 50 /data/zhaojing/marble/tensor/dataset1/subs_splittrain_1.csv /data/zhaojing/marble/tensor/dataset1/vals_binary_splittrain_1.csv /data/zhaojing/marble/tensor/siz.csv 150 /data/zhaojing/marble/tensor/dataset1/subs_splittest_1.csv /data/zhaojing/marble/tensor/dataset1/vals_binary_splittest_1.csv /data/zhaojing/marble/tensor/siz.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_S1_50_2.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_S1_50_2.txt > /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/redirect_S1_50_2 2>&1 &
echo "ssh logbase-a05"
ssh logbase-a06 nohup python /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/marbleTests_zj_argu.py 50 /data/zhaojing/marble/tensor/dataset1/subs_splittrain_1.csv /data/zhaojing/marble/tensor/dataset1/vals_binary_splittrain_1.csv /data/zhaojing/marble/tensor/siz.csv 150 /data/zhaojing/marble/tensor/dataset1/subs_splittest_1.csv /data/zhaojing/marble/tensor/dataset1/vals_binary_splittest_1.csv /data/zhaojing/marble/tensor/siz.csv /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_train_S1_50_3.txt /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/membership_test_S1_50_3.txt > /home/singa/zhaojing/NUH-singa/incubator-singa/examples/data-process/python/Marble-code/membershipresult/redirect_S1_50_3 2>&1 &
echo "ssh logbase-a06" 