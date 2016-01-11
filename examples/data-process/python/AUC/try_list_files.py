import os

def Test1(rootDir): 
    list_dirs = os.walk(rootDir) 
    arr = []
    for root, dirs, files in list_dirs: 
        for d in dirs:
            print "ds" 
            print os.path.join(root, d)      
        for f in files:
            # print "fs" 
            print os.path.join(root, f)
            arr.append(os.path.join(root, f))
    for i in range(len(arr)):
        print "in arr file = \n", arr[i]
     

Test1('/data/zhaojing/AUC/label/')
