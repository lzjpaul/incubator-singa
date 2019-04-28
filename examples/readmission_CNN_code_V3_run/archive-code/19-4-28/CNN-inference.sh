echo $1
echo $2
echo $3
echo $4
echo "python model/readmission_CNN_code/CNN-readmission-inference.py -inputfolder $1 -outputfolder $2 -visfolder $3 -sampleid $4 --use_cpu"
python model/readmission_CNN_code/CNN-readmission-inference.py -inputfolder $1 -outputfolder $2 -visfolder $3 -sampleid $4 --use_cpu
