import numpy as np

doc_num = 3
word_num = 4
topic_num = 3
phi = np.array([[0.0,0.0,0.0],[0.2,0.3,0.4],[0.0,0.0,0.0],[0.4,0.5,0.6]])
theta_alldoc = np.array([[1/3., 1/3., 1/3.],[1/3., 1/3., 1/3.],[1/3., 1/3., 1/3.]])
# calc the resposibilities for pj(wi)
responsibility_all_doc = np.zeros((doc_num, word_num, topic_num))
for doc_idx in range(doc_num):
    responsibility_doc = phi*(theta_alldoc[doc_idx].reshape((1, -1)))
    # responsibility normalized with summation(denominator)
    responsibility_all_doc[doc_idx] = responsibility_doc/(np.sum(responsibility_doc, axis=1).reshape(-1,1))
    # for some words, the topic distributions are all zeros
    zero_idx = np.where((np.sum(responsibility_doc, axis=1)) == 0)[0]
    average_theta_matrix = np.full((len(zero_idx), topic_num), 1./topic_num)
    responsibility_all_doc[doc_idx][zero_idx] = average_theta_matrix
    print ("doc_idx: ", doc_idx)
    print ("responsibility_doc: ", responsibility_doc)
    print ("responsibility_all_doc[doc_idx]: ", responsibility_all_doc[doc_idx])
    print ("word responsibility sum: ", np.sum(responsibility_all_doc[doc_idx], axis=1))
print ('responsibility_all_doc shape: ', responsibility_all_doc.shape)
print ('responsibility_all_doc: ', responsibility_all_doc)
print ('responsibility_all_doc sum: ', np.sum(responsibility_all_doc))
'''
>>> np.sort(np.unique(phi.reshape((1,-1))))
array([  0.00000000e+00,   6.90055773e-08,   6.90055773e-08, ...,
         3.50406289e-01,   4.21247430e-01,   4.62245913e-01])
'''
