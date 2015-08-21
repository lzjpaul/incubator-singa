mkdir $1
mkdir $2
./create_shard.bin $3 $4 $5 $6 $7 $8
./create_shard.bin $9 ${10} ${11} ${12} ${13} ${14}
/home/singa/zhaojing/incubator-singa/bin/singa-run.sh -model=${15} -cluster=${16}
