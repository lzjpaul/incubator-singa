import numpy as np

doc_num = 3
word_num = 4
topic_num = 3
responsibility_all_doc = np.array([[[0.1,0.2,0.3],[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4,0.5,0.6]], [[0.1,0.2,0.3],[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4,0.5,0.6]], [[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4,0.5,0.6],[0.5,0.6,0.7]]])
print 'responsibility_all_doc shape: ', responsibility_all_doc.shape
# calc the resposibilities for pj(wi)
w_array = np.array([[-1.,2.,-3.],[4.,-5.,6.],[7.,-8.,9.],[-10.,11.,-12.]])

theta_alldoc = np.zeros((doc_num, topic_num))
alpha = 1.05
# update theta_all_doc
for doc_idx in range(doc_num):
    print 'doc_idx: ', doc_idx
    theta_doc = (np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (alpha - 1)) / np.sum(np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (alpha - 1))
    print '(responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))) \n', (responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1)))
    print '(np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0)) \n', (np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0))
    print '(np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (alpha - 1)): \n', (np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (alpha - 1))
    print 'np.sum(np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (alpha - 1))', np.sum(np.sum((responsibility_all_doc[doc_idx] * np.absolute(w_array[:, doc_idx]).reshape((-1,1))), axis=0) + (alpha - 1))
    print 'theta_doc: ', theta_doc
    theta_alldoc[doc_idx] = theta_doc
print 'sum: np.sum(theta_alldoc,axis=1): ', np.sum(theta_alldoc,axis=1)
print 'sum: np.sum(theta_alldoc): ', np.sum(theta_alldoc)


