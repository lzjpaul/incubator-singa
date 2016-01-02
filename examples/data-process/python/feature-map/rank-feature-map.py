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
group_length = 14

print "width = \n", width
print "sample_num = \n", sample_num
print "filter_num = \n", filter_num

filter_fea_value_vec = np.zeros(topk * sample_num)
filter_fea_position_vec = np.zeros(topk * sample_num)
all_filter_top_position = np.zeros([filter_num,topk])

# array sort http://blog.csdn.net/maoersong/article/details/21875705
for i in range(filter_num):
    # print "finish filter_num: ", i
    for j in range(sample_num):
        filter_fea_value_vec[(j * topk):(j * topk + topk)] = featuremap_matrix[(j * filter_num * topk + i * topk): (j * filter_num * topk + i * topk + topk),1]
	filter_fea_position_vec[(j * topk):(j * topk + topk)] = featuremap_matrix[(j * filter_num * topk + i * topk): (j * filter_num * topk + i * topk + topk),0]
    all_filter_top_position[i,:] = filter_fea_position_vec[np.argsort(-filter_fea_value_vec)][0:topk]
    if i == 0:
        print "top position of i == 0 = \n ", all_filter_top_position[i,:]
        print "ffilter_fea_position_vec = \n", filter_fea_position_vec

all_filter_top_position_mod = all_filter_top_position % width
all_filter_top_position_mod_div = all_filter_top_position_mod * (1/float(group_length))
#output
a = numpy.asarray(all_filter_top_position, dtype = int)
numpy.savetxt(sys.argv[5], a, fmt = '%d', delimiter=",") #original output
b = numpy.asarray(all_filter_top_position_mod, dtype = int)
numpy.savetxt(sys.argv[6], b, fmt = '%d', delimiter=",")
c = numpy.asarray(all_filter_top_position_mod_div, dtype = int)
numpy.savetxt(sys.argv[7], c, fmt = '%d', delimiter=",")
# python rank-feature-map.py /data/zhaojing/feature-map/shape/version200.csv /data/zhaojing/feature-map/map/version200.csv 30 10 /data/zhaojing/feature-map/version200-output.csv /data/zhaojing/feature-map/version200-output-mod.csv /data/zhaojing/feature-map/version200-output-mod-div.csv
