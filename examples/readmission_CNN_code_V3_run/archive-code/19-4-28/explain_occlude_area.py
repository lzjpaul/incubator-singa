import sys, os
import traceback
import time
import urllib
import numpy as np
from argparse import ArgumentParser


# from singa import tensor, device, optimizer
# from singa import utils
# from singa.proto import core_pb2
# from rafiki.agent import Agent, MsgType

# import model

import pdb

def explain_occlude_area(test_feature, test_label, probpath, truelabelprobpath, metadatapath, top_n):
    probmatrix = np.genfromtxt(probpath, delimiter=',', dtype=str)
    truelabelprobmatrix = np.genfromtxt(truelabelprobpath, delimiter=',', dtype=np.float32)
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
    # check the shape
    for n in range(test_feature.shape[0]):
        # for this specific patient, which features are non-zero
        sample_index_matrix = test_feature[n].reshape((height, width)) * index_matrix.reshape((height, width))
        print "sample: ", probmatrix[n, 0]
        if test_label is not None:
            print "label: ", test_label[n]
        print "readmitted prob n: ", probmatrix[n, 1]
        print "non zero index: ", np.nonzero(sample_index_matrix)
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
    
    # test_feature = np.genfromtxt(args.featurepath, dtype=np.float32, delimiter=',')
    # test_label = np.genfromtxt(args.labelpath, dtype=np.int32, delimiter=',')
    # probmatrix = np.genfromtxt(args.probpath, delimiter=',',)
    # truelabelprobmatrix = np.genfromtxt(args.truelabelprobpath, delimiter=',')
    # meta_data = np.genfromtxt(args.metadatapath, delimiter=',')
    args.probpath = 'outputfolder/readmitted_prob.csv'
    args.truelabelprobpath = 'outputfolder/true_label_prob_matrix.csv'
    args.metadatapath = 'outputfolder/meta_data.csv'

    # explain_occlude_area(test_feature, test_label, args.probpath, args.truelabelprobpath, args.metadatapath, top_n = 30)
    explain_occlude_area(None, None, args.probpath, args.truelabelprobpath, args.metadatapath, top_n = 30)


if __name__ == '__main__':
    main()

# python explain_occlude_area.py -featurepath /data/zhaojing/regularization/LACE-CNN-1500/reverse-order/nuh_fa_readmission_case_demor_inpa_kb_ordered_output_onehot_12slots_reverse.csv -labelpath /data/zhaojing/regularization/LACE-CNN-1500/nuh_fa_readmission_case_label.csv -probpath readmitted_prob.csv -truelabelprobpath true_label_prob_matrix.csv -metadatapath meta_data.csv
