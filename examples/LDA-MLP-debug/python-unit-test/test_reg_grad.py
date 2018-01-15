import numpy as np
doc_num = 3
word_num = 4
topic_num = 3
phi = np.array([[0.0,0.0,0.0],[0.2,0.3,0.4],[0.0,0.0,0.0],[0.4,0.5,0.6]])
theta_alldoc = np.array([[1/3., 1/3., 1/3.],[1/3., 1/3., 1/3.],[1/6., 1/6., 2/3.]])
w_array = np.array([[-1.,2.,-3.],[4.,-5.,6.],[7.,-8.,9.],[-10.,11.,-12.]])
print ('w_array: ', w_array)
theta_phi_all_doc = np.zeros((word_num, doc_num))

for doc_idx in range(doc_num):
    print ('doc_idx: ', doc_idx)
    theta_phi_doc = phi*(theta_alldoc[doc_idx].reshape((1, -1)))
    print ('after phi multiplication theta_phi_doc: ', theta_phi_doc)
    theta_phi_doc = np.sum(theta_phi_doc, axis=1)
    zero_idx = np.where(theta_phi_doc == 0)[0]
    print ('after sum theta_phi_doc: ', theta_phi_doc)
    min_theta_phi_doc = np.full((len(zero_idx),), -100.)
    theta_phi_doc = np.log(theta_phi_doc)
    print ('after log theta_phi_doc: ', theta_phi_doc)
    theta_phi_doc[zero_idx] = min_theta_phi_doc
    print ('after filling -100 log theta_phi_doc: ', theta_phi_doc)
    theta_phi_all_doc[:, doc_idx] = theta_phi_doc
print ('theta_phi_all_doc: ', theta_phi_all_doc)
print ('np.sort(np.unique(theta_phi_all_doc.reshape((1,-1)))): ', np.sort(np.unique(theta_phi_all_doc.reshape((1,-1)))))
print ('return -(np.sign(w_array) * theta_phi_all_doc):', (-(np.sign(w_array) * theta_phi_all_doc)))
    
'''
min: np.sort(np.unique(theta_phi_all_doc.reshape((1,-1)))):  [        -inf -15.53806682 -15.53790758 ...,  -2.78401928  -2.78262487
  -2.78043074]
'''
