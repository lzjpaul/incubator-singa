CUDA_VISIBLE_DEVICES=1 python CNN-readmission-trainvalidationtest-wd01.py -inputfolder ../saved_sparse_data/ -outputfolder 'outputfolder' -visfolder 'visfolder' --max_epoch 100 | tee -a 19-4-27-check-wd-convergence-4.log