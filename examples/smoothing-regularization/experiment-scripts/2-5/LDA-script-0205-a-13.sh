python /home/singa/zhaojing/smooth-regularization-dbpcm-bak-logistic/singa/incubator-singa/examples/smoothing-regularization/Example-smoothing-regularization-main-all-param.py -datapath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_lastcase.csv -labelpath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv -clf ridge -labelcolumn 1 -batchsize 30 -svmlight 0 -sparsify 0 -scale 0 -batchgibbs 0 | tee -a /data/zhaojing/regularization/log0205/02-05-9.log