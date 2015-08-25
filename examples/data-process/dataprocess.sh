echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo ${10}
echo ${11}
python python/standardization.py $1 $2 $3 $4
python python/splittraintest.py $5 $6 $7 $8 $9 ${10} ${11} ${12}
#python $1 $2 $3 $4 $5 $6 $7 $8 $9
#./dataprocess.sh 15(fea_num) data/test_script_traintest.csv data/test_script_traintest_data_norm.csv data/test_script_traintest_label.csv
# 18(train_end rownum) 28(test_end rownum) data/test_script_traintest_data_norm.csv(data input) data/test_script_traintest_label.csv(label input)
#data/test_script_train_data_norm.csv(train_data) data/test_script_train_label.csv(train_label) data/test_script_test_data_norm.csv(test_data) data/test_script_test_label.csv(test_label)
