import json
import csv
import numpy as np

f = open("/data/zhaojing/proc2Levelstry.json")
x = json.load(f)
f = csv.writer(open("/data/zhaojing/testjson.csv", "wb+"))
print x
#for x in x:
#    f.writerow(x["type"],x["desc"],x["L2"],x["L3"],x["L1"]])

file = open("/data/zhaojing/proc2UMLS-txt.txt")
json_file = np.genfromtxt(file, delimiter=",")
a = file.readlines()
print len(a)
print a
file.close()


print json_file.shape

file = open("/data/zhaojing/testjson.txt", "w")
np.savetxt(file, json_file, "%s", ",")
file.close()
