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
