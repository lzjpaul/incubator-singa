echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo "python model/readmission_CNN_code/CNN-readmission-trainvalidationtest.py -inputfolder $1 -outputfolder $2 -visfolder $3 -trainratio $4 -validationratio $5 -testratio $6 --use_cpu"
python model/readmission_CNN_code/CNN-readmission-trainvalidationtest.py -inputfolder $1 -outputfolder $2 -visfolder $3 -trainratio $4 -validationratio $5 -testratio $6 --use_cpu
