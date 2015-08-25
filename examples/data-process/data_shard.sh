mkdir $1
mkdir $2
./create_shard.bin $3 $4 $5 $6 $7 $8
./create_shard.bin $9 ${10} ${11} ${12} ${13} ${14}
#/home/singa/zhaojing/incubator-singa/bin/singa-run.sh -model=${15} -cluster=${16}
#./run-singa.sh ../data-process/data/test_script_train_shard(folder) ../data-process/data/test_script_test_shard(folder) 18(num_item) 1(row) 15(column) data/test_script_train_data_norm.csv data/test_script_train_label.csv
#data/test_script_train_shard/(folder) 10 1 15 data/test_script_test_data_norm.csv data/test_script_test_label.csv data/test_script_test_shard/
