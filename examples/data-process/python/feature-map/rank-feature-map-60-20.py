# !!!attention!!! -- step == 0 also outputs, so only collect step == 1000
# differs in how to calculate group line 91


# group_length change to sys.argv (over)
# how to check the end of images? -- print line 56: check the last one and first one is zero? check sort.. (over)
# correct???  : batchsize --> filter_num(how to reorganize each filter) --> height --> width (mod 114) (over)
# when read in like: height * width: mod width correct (different picture and different filter the same arg height * width) (over)
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
file = open(sys.argv[1])
shape_matrix = np.genfromtxt(file, delimiter=",")
shape_matrix = shape_matrix.astype(np.int)
file.close()
print "shape_matrix =\n", shape_matrix

file = open(sys.argv[2])
featuremap_matrix_origin = np.genfromtxt(file, delimiter=",")
file.close()
print "featuremap_matrix_origin shape =\n", featuremap_matrix_origin.shape

tol_length = len(featuremap_matrix_origin[:,0])
print "tol_length = \n", tol_length
print "tol_length/2 = \n", tol_length/2
#only reserve the latter half
featuremap_matrix = featuremap_matrix_origin[tol_length/2 : tol_length, :]
print "featuremap_matrix shape =\n", featuremap_matrix.shape

batchsize = shape_matrix[0][0]
test_steps = int (sys.argv[3]) # 30
sample_num = batchsize * test_steps
filter_num = shape_matrix[0][1]
height = shape_matrix[0][2]
width = shape_matrix[0][3]
topk = int (sys.argv[4]) # 10
group_length = int (sys.argv[5]) # 14

print "width = \n", width
print "sample_num = \n", sample_num
print "filter_num = \n", filter_num
print "height = \n", height
print "width = \n", width

filter_fea_value_vec = np.zeros(topk * sample_num)
filter_fea_position_vec = np.zeros(topk * sample_num)
all_filter_top_position = np.zeros([filter_num,topk])

# array sort http://blog.csdn.net/maoersong/article/details/21875705
for i in range(filter_num):
    # print "finish filter_num: ", i
    for j in range(sample_num):
        filter_fea_value_vec[(j * topk):(j * topk + topk)] = featuremap_matrix[(j * filter_num * topk + i * topk): (j * filter_num * topk + i * topk + topk),1]
	filter_fea_position_vec[(j * topk):(j * topk + topk)] = featuremap_matrix[(j * filter_num * topk + i * topk): (j * filter_num * topk + i * topk + topk),0]
    all_filter_top_position[i,:] = filter_fea_position_vec[np.argsort(-filter_fea_value_vec)][0:topk] # how to check this one?
    if i == 0:
        print "top position index of i == 0 = \n ", np.argsort(-filter_fea_value_vec)[0:topk]
        # print "ffilter_fea_position_vec = \n", filter_fea_position_vec
        try_value = numpy.asarray(filter_fea_value_vec, dtype = float)
        numpy.savetxt("/data/zhaojing/feature-map/try_value.csv", try_value, fmt = '%6f', delimiter=",") #original output
        try_position = numpy.asarray(filter_fea_position_vec, dtype = int)
        numpy.savetxt("/data/zhaojing/feature-map/try_position.csv", try_position, fmt = '%d', delimiter=",") #original output
    if i == 0:
        print "0: value 1 = \n", filter_fea_value_vec[1]
        print "0: position 1 = \n", filter_fea_position_vec[1]
        print "0: value 13 = \n", filter_fea_value_vec[13]
        print "0: position 13 = \n", filter_fea_position_vec[13]
        print "0: value 24 = \n", filter_fea_value_vec[24]
        print "0: position 24 = \n", filter_fea_position_vec[24]
        print "0: value 29998 = \n", filter_fea_value_vec[29998]
        print "0: position 29998 = \n", filter_fea_position_vec[29998]
    if i == (filter_num - 1):
        print "499: value 1 = \n", filter_fea_value_vec[1]
        print "499: position 1 = \n", filter_fea_position_vec[1]
        print "499: value 13 = \n", filter_fea_value_vec[13]
        print "499: position 13 = \n", filter_fea_position_vec[13]
        print "499: value 24 = \n", filter_fea_value_vec[24]
        print "499: position 24 = \n", filter_fea_position_vec[24]
        print "499: value 29998 = \n", filter_fea_value_vec[29998]
        print "499: position 29998 = \n", filter_fea_position_vec[29998]


all_filter_top_position_mod = all_filter_top_position % width
all_filter_top_position_mod_div = (all_filter_top_position_mod - 2) * (1/float(group_length)) + 1
#output
a = numpy.asarray(all_filter_top_position, dtype = int)
numpy.savetxt(sys.argv[6], a, fmt = '%d', delimiter=",") #original output
b = numpy.asarray(all_filter_top_position_mod, dtype = int)
numpy.savetxt(sys.argv[7], b, fmt = '%d', delimiter=",")
c = numpy.asarray(all_filter_top_position_mod_div, dtype = int)
numpy.savetxt(sys.argv[8], c, fmt = '%d', delimiter=",")
# python examples/data-process/python/feature-map/rank-feature-map-60-20.py /data/zhaojing/feature-map/shape/version691.csv /data/zhaojing/feature-map/map/version691.csv 30 10 3 /data/zhaojing/feature-map/version691-output.csv /data/zhaojing/feature-map/version691-output-mod.csv /data/zhaojing/feature-map/version691-output-mod-div.csv
