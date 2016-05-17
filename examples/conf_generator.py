import numpy as np
import random
from random import randint
import numpy
import os
import sys
import shutil

for i in range(100):
    f = open("MLP_NUHALLCOND_1_visit_script.sh", 'a+b')
    f.write("python NUHALLCOND_DIAG_VISIT_1.py /data1/zhaojing/NUHALLCOND/VISIT_DIAG_aggcnt_1/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_train_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/VISIT_DIAG_aggcnt_1/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_train_label_normrange_1.csv /data1/zhaojing/NUHALLCOND/VISIT_DIAG_aggcnt_1/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_test_data_normrange_1.csv /data1/zhaojing/NUHALLCOND/VISIT_DIAG_aggcnt_1/subsample1/NUH_DS_SOC_READMIT_DIAG_LAB_INOUTPATIENT_CNN_SAMPLE_DIAG_1src_KB_VISIT_DIAG_aggcnt_1_test_label_normrange_1.csv > redirect/NUH_MLP/version" + str(i) + "\n")
f.close()
# python conf_generator.py 3-6-KB-adaptive-diff-filters-nodrop-multigroup-400.conf multigroup400
