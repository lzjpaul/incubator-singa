import numpy as np

file = open("/data/zhaojing/marble/mlp-samples/Marble_split_trainindex_1.csv")
index = np.genfromtxt(file, delimiter=",")
file.close()
print "index.shape =\n", index.shape

index = index.astype(np.int)
print "index = \n", index

print "find 1 = \n", 1 in index
print "find 5 = \n", 5 in index
print "find 11 = \n", 11 in index
print "find 12 = \n", 12 in index

