# check: some cases are all 0
import numpy as np
import numpy
import sys
#from sklearn.preprocessing import OneHotEncoder

#read data
def readData(filename):
    return [line.rstrip().split(',') for line in open(filename)]

#read training data
lab_origin_data = readData(sys.argv[1])  #modify here
lab_origin_data_matrix = np.array(lab_origin_data[0:])[:,0:]
lab_origin_data_matrix = lab_origin_data_matrix.astype(np.int)
print "lab_origin_data_matrix shape"
print lab_origin_data_matrix.shape

# how many cases for each sample ..
case_num = int (sys.argv[2])
print "first row = \n", lab_origin_data_matrix[0,:]
print "origin data length = \n", len(lab_origin_data_matrix[0,:])
case_length = (len(lab_origin_data_matrix[0,:])/case_num)
# H, L, N, Unknown
lab_num = case_length/4
print "sample_length = \n", len(lab_origin_data_matrix[0,:])

lab_laplacian_data_matrix = np.zeros([len(lab_origin_data_matrix[:,0]),len(lab_origin_data_matrix[0,:])])
print "lab_laplacian_data_matrix shape"
print lab_laplacian_data_matrix.shape

case_lab_tol = np.zeros(case_length)

for i in range(lab_num):
    H_tol = 0;
    L_tol = 0;
    N_tol = 0;
    Unknown_tol = 0;
    for j in range(case_num):
	H_tol = H_tol + sum(lab_origin_data_matrix[:, case_length * j + 4 * i])
	L_tol = L_tol + sum(lab_origin_data_matrix[:, case_length * j + 4 * i + 1])
	N_tol = N_tol + sum(lab_origin_data_matrix[:, case_length * j + 4 * i + 2])
	Unknown_tol = Unknown_tol + sum(lab_origin_data_matrix[:, case_length * j + 4 * i + 3])
    case_lab_tol[4 * i] = H_tol
    case_lab_tol[4 * i + 1] = L_tol
    case_lab_tol[4 * i + 2] = N_tol
    case_lab_tol[4 * i + 3] = Unknown_tol

case_lab_mean = np.zeros(case_length)
# how many are non-zero cases for this specific lab ... check!!!
for i in range(lab_num):
    non_zero_cases = 0;
    # traverse all the samples
    for j in range(len(lab_origin_data_matrix[:,0])):
	for k in range(case_num):
	    if sum(lab_origin_data_matrix[j, (case_length * k + 4 * i):(case_length * k + 4 * i + 4)]) > 0:
		non_zero_cases = non_zero_cases + 1
    print "lab index i = \n ", i
    print "non_zero_cases = \n", non_zero_cases
    if non_zero_cases > 0:
        case_lab_mean[4 * i] = case_lab_tol[4 * i] / non_zero_cases
        case_lab_mean[4 * i + 1] = case_lab_tol[4 * i + 1] / non_zero_cases
        case_lab_mean[4 * i + 2] = case_lab_tol[4 * i + 2] / non_zero_cases
        case_lab_mean[4 * i + 3] = case_lab_tol[4 * i + 3] / non_zero_cases
    elif non_zero_cases == 0:
        print "H tol = \n", case_lab_tol[4 * i]
        print "L tol = \n", case_lab_tol[4 * i + 1]
        print "N tol = \n", case_lab_tol[4 * i + 2]
        print "Unknown tol = \n", case_lab_tol[4 * i + 3]
        case_lab_mean[4 * i] = 0
        case_lab_mean[4 * i + 1] = 0
        case_lab_mean[4 * i + 2] = 0
        case_lab_mean[4 * i + 3] = 0

# traverse all the samples
# check is it float?
for j in range(len(lab_origin_data_matrix[:,0])):
    for k in range(case_num):
	for i in range(lab_num):
	    sample_case_lab_tol = 0
	    sample_case_lab_tol = sum(lab_origin_data_matrix[j, (case_length * k + 4 * i):(case_length * k + 4 * i + 4)]) + sample_case_lab_tol # check correct ??? w + x +y + z
	    sample_case_lab_tol = sum(case_lab_mean[(4 * i) : (4 * i + 4)]) + sample_case_lab_tol # a + b + c + d + w + x +y + z
            if sample_case_lab_tol > 0:
	        lab_laplacian_data_matrix[j, (case_length * k + 4 * i)] = (lab_origin_data_matrix[j, (case_length * k + 4 * i)] + case_lab_mean[4 * i]) / sample_case_lab_tol
	        lab_laplacian_data_matrix[j, (case_length * k + 4 * i + 1)] = (lab_origin_data_matrix[j, (case_length * k + 4 * i + 1)] + case_lab_mean[4 * i + 1]) / sample_case_lab_tol
	        lab_laplacian_data_matrix[j, (case_length * k + 4 * i + 2)] = (lab_origin_data_matrix[j, (case_length * k + 4 * i + 2)] + case_lab_mean[4 * i + 2]) / sample_case_lab_tol
	        lab_laplacian_data_matrix[j, (case_length * k + 4 * i + 3)] = (lab_origin_data_matrix[j, (case_length * k + 4 * i + 3)] + case_lab_mean[4 * i + 3]) / sample_case_lab_tol
            elif sample_case_lab_tol == 0:
                lab_laplacian_data_matrix[j, (case_length * k + 4 * i)] = 0
                lab_laplacian_data_matrix[j, (case_length * k + 4 * i + 1)] = 0
                lab_laplacian_data_matrix[j, (case_length * k + 4 * i + 2)] = 0
                lab_laplacian_data_matrix[j, (case_length * k + 4 * i + 3)] = 0

#output
a = numpy.asarray(lab_laplacian_data_matrix, dtype = float)
numpy.savetxt(sys.argv[3], a, fmt = '%6f', delimiter=",") #modify here
# python lab_laplacian.py lab_original_data.csv 2 lab_laplacian_data.csv
