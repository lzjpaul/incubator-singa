python /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/singa/incubator-singa/examples/smoothing-regularization/Example-smoothing-regularization-main-all-param.py -datapath /data/zhaojing/regularization/uci-dataset/uci-diabetes-readmission/diag-dim-reduction/diabetic_data_diag_low_dim_2_class_categorical.csv -clf gaussianmixture -labelcolumn 1 -batchsize 1000 -svmlight 0 -sparsify 0 -scale 0 -batchgibbs 0 | tee -a /data/zhaojing/regularization/log0205/02-05-1.log