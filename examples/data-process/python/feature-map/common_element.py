import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
file = open(sys.argv[1])
filters = np.genfromtxt(file, delimiter=",")
filters_matrix = filters.astype(np.int)
file.close()
print "filters_matrix shape =\n", filters_matrix.shape

filters_common_area = np.zeros(len(filters_matrix[:,0]))
for i in range(len(filters_matrix[:,0])):
    counts = np.bincount(filters_matrix[i,:])
    filters_common_area[i] = np.argmax(counts)
a = numpy.asarray(filters_common_area, dtype = int)
numpy.savetxt(sys.argv[2], a, fmt = '%d', delimiter=",") #original output
# python examples/data-process/python/feature-map/common_element.py /data/zhaojing/feature-map/version528-output-mod-div.csv /data/zhaojing/feature-map/version528-common-area-mod-div.csv
