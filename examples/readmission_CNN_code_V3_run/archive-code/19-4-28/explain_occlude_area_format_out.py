import sys, os
import traceback
import time
import urllib
import numpy as np
from argparse import ArgumentParser


from singa import tensor, device, optimizer
from singa import utils
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType

import model
import json
import pdb
from collections import OrderedDict
import cPickle
# the information is output according to standardformat
# the indices of generating json is deduplicated
# the features are ranked features

def explain_occlude_area_format_out(visfolder, test_feature, test_label, probpath, truelabelprobpath, metadatapath, patientdrgpath, drgpath, top_n):
    probmatrix = np.genfromtxt(probpath, delimiter=',', dtype=str)
    truelabelprobmatrix = np.genfromtxt(truelabelprobpath, delimiter=',')
    meta_data = np.genfromtxt(metadatapath, delimiter=',')
    height_dim, height, kernel_y, stride_y, width_dim, width, kernel_x, stride_x = \
    int(meta_data[0]), int(meta_data[1]), int(meta_data[2]), int(meta_data[3]), int(meta_data[4]), int(meta_data[5]), int(meta_data[6]), int(meta_data[7])
    print "meta_data: ", meta_data
    print "truelabelprobmatrix shape: ", truelabelprobmatrix.shape
    print "probmatrix shape: ", probmatrix.shape
    # print "test_feature shape: ", test_feature.shape
    top_n_array = truelabelprobmatrix.argsort()[0:top_n]
    print "top_n_array: ", top_n_array
    print "top_n_array % width_dim: ", (top_n_array % width_dim)
    index_matrix = np.zeros(height * width)
    print "index_matrix shape: ", index_matrix.shape
    print "top_n: ", top_n
    with open(patientdrgpath, 'rb') as f:
        patient_drg_dict = cPickle.load(f)
    with open(drgpath, 'rb') as f:
        drg_dict = cPickle.load(f)
    # !!! correcttt ??? -- quyu
    # step 1: which areas of feature map (height_idx, width_idx)
    # step 2: which corresponding features in the original data matrix (index_matrix)
    # step 3: for each sample, which are significant risk factors? test_feature[n] * index_matrix
    for i in range(top_n_array.shape[0]):
        height_idx = int(top_n_array[i] / width_dim)
        width_idx = int(top_n_array[i] % width_dim)
        for j in range (int(kernel_y)):
            # the features of significant features are non-zero
            index_matrix[((height_idx * stride_y + j) * width + width_idx * stride_x) : ((height_idx * stride_y + j) * width + width_idx * stride_x + kernel_x)] = float(1.0)
    print "index_matrix sum: ", index_matrix.sum()
    index_matrix = index_matrix.reshape((height, width))
    ### hard-code LOS and DRG ###
    index_matrix[11, 146:153] = 1 # last visit LOS
    index_matrix[:, 319:567] = 1 # DRG
    ### delete the following features ###
    index_matrix[:, 17:121] = 0 # Nationality and Race
    index_matrix[:, 121:129] = 0 # MartialStatus
    # index_matrix[:, 140:149] = 0 # Number_of_SOC_Visits
    ### only last visit shows the following information (visit level information except DRG)
    index_matrix[0:11, 129:319] = 0
    print "index_matrix shape: ", index_matrix.shape
    print "index_martix: "
    feature_explanation = np.genfromtxt('readmission-feature-mapping-explanation.txt', delimiter='\t', dtype=str)
    feature_explanation = feature_explanation[:, 0]
    print np.concatenate((np.nonzero(index_matrix)[0].reshape((-1,1)), np.nonzero(index_matrix)[1].reshape((-1,1))), axis=1)
    print np.concatenate((np.nonzero(index_matrix)[0].reshape((-1,1)).astype(str), feature_explanation[np.nonzero(index_matrix)[1].reshape((-1,1))]), axis=1)
    
    # check the shape
    for n in range(test_feature.shape[0]):
        # for this specific patient, which features are non-zero
        sample_index_matrix = test_feature[n].reshape((height, width)) * index_matrix
        print "sample: ", probmatrix[n, 0]
        if test_label is not None:
            print "label: ", test_label[n]
        print "readmitted-prob: ", probmatrix[n, 1]

        print np.concatenate((np.nonzero(sample_index_matrix)[0].reshape((-1, 1)), np.nonzero(sample_index_matrix)[1].reshape((-1, 1))), axis=1)
        print np.concatenate((np.nonzero(sample_index_matrix)[0].reshape((-1, 1)).astype(str), feature_explanation[np.nonzero(sample_index_matrix)[1].reshape((-1, 1))]), axis=1)

        sample_info_dict = OrderedDict() # for output to json
        sample_info_dict['patient mrn'] = probmatrix[n, 0]
        sample_info_dict['result'] = round(float(probmatrix[n, 1]), 2)
        
        feature_list = []
        # add DRG features into feature_list
        drgs = patient_drg_dict[probmatrix[n, 0]]['DRG'].split("|")
        print drgs
        for drg in drgs:
            if drg != "" and drg != "-":
                try:
                    desc = drg_dict[drg]["DESCRIPTION"]
                    feature_list.append("DRG: %s (%s)" % (drg, desc))
                    with open(os.path.join(visfolder, 'record_patient.txt'), 'a') as f:
                        f.write(probmatrix[n, 0] + '\n')
                except:
                    feature_list.append("DRG: %s" % drg)
        feature_idx_array = list(set(np.nonzero(sample_index_matrix)[1]))
        
        # sort the feature_idx_array according to ordered features
        ranked_all_features_idx = np.genfromtxt('ranked_readmission_feature_mapping_explanation.txt', delimiter='\t', dtype=str)[:, 2].astype(np.int)
        # map to a ranked feature index (according to significance) table
        ranked_feature_idx = ranked_all_features_idx[feature_idx_array]
        ranked_feature_idx_sort = np.argsort(ranked_feature_idx)
        feature_idx_array = np.array(feature_idx_array)[ranked_feature_idx_sort]
        print "sorted feature_idx_array: ", feature_idx_array
        # add other information into feature _list
        for feature_idx in feature_idx_array:
            if feature_idx > 318:
                continue
            detail = feature_explanation[feature_idx]
            if detail.split(':')[1].strip() == "":  # empty value in database
                continue
            feature_list.append(detail)
        print feature_list

        sample_info_dict['result_description'] = {"content": feature_list}
        sample_info_dict['predictor'] = 'readm'
        sample_info_dict['dt_predicted'] = 'time()'
        json_file = os.path.join(visfolder, str(probmatrix[n, 0])+'.json')
        try:
            with open(json_file, 'w') as sample_info_writer:
                json.dump(sample_info_dict, sample_info_writer)    
        except Exception as e:
            os.remove(json_file)
            print('output patient json failed: ', e)
        print "\n"
            

def main():
    '''Command line options'''
    # Setup argument parser
    parser = ArgumentParser(description="Train CNN Readmission Model")

    parser.add_argument('-featurepath', type=str, help='the test feature path')
    parser.add_argument('-labelpath', type=str, help='the test label path')
    parser.add_argument('-probpath', type=str, help='the prob path')
    parser.add_argument('-truelabelprobpath', type=str, help='the true label prob path')
    parser.add_argument('-metadatapath', type=str, help='the meta data path')
        
    # Process arguments
    args = parser.parse_args()

    inputfolder = '../saved_data'
    visfolder = 'visfolder'
    outputfolder = 'outputfolder'
    probpath = os.path.join(outputfolder, 'readmitted_prob.csv')
    truelabelprobpath = os.path.join(outputfolder,'true_label_prob_matrix.csv')
    metadatapath = os.path.join(outputfolder,'meta_data.csv')
    patientdrgpath = os.path.join(inputfolder, 'patient_DRG_info.pkl')
    drgpath = os.path.join(inputfolder, 'DRG_dict.pkl')
    top_n = 30
        
    # test_feature = np.genfromtxt(args.featurepath, delimiter=',')
    # test_label = np.genfromtxt(args.labelpath, delimiter=',')
    test_feature = None
    test_label = None
    # probmatrix = np.genfromtxt(args.probpath, delimiter=',')
    # truelabelprobmatrix = np.genfromtxt(args.truelabelprobpath, delimiter=',')
    # meta_data = np.genfromtxt(args.metadatapath, delimiter=',')

    # explain_occlude_area(test_feature, test_label, args.probpath, args.truelabelprobpath, args.metadatapath, top_n = 3)
    explain_occlude_area_format_out(visfolder, test_feature, test_label, probpath, truelabelprobpath, metadatapath, patientdrgpath, drgpath, top_n)

if __name__ == '__main__':
    main()

# python explain_occlude_area.py -featurepath /data/zhaojing/regularization/LACE-CNN-1500/reverse-order/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_12slots_reverse.csv -labelpath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv -probpath readmitted_prob.csv -truelabelprobpath true_label_prob_matrix.csv -metadatapath meta_data.csv
