import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

def cal_accuracy(yPredict, yTrue):
    return np.sum(((yPredict > 0.5) == yTrue).astype(int)) / float(yTrue.shape[0])

def auroc(yPredictProba, yTrue):
    return roc_auc_score(yTrue, yPredictProba)


def HealthcareMetrics(y_prob, y_true, prob_threshold):
    tp_idx = np.where(y_true==1)[0]
    fp_idx = np.where(y_true==0)[0]
    true_1 = len(tp_idx)
    true_0 = len(fp_idx)
    predict_1 = len(np.where(y_prob>=prob_threshold)[0])
    predict_0 = len(np.where(y_prob<prob_threshold)[0])
    correct_1 = len(np.where(y_prob[tp_idx]>=prob_threshold)[0])
    correct_0 = len(np.where(y_prob[fp_idx]<prob_threshold)[0])
    # print "Metric Begin"
    try:
        precision_1 = correct_1 / float(predict_1)
    except:
        precision_1 = 0
    try:
        recall_1 = correct_1 / float(true_1)
    except:
        recall_1 = 0
    try:
        Fmeasure_1 = 2*precision_1*recall_1 / float(precision_1 + recall_1)
    except:
        Fmeasure_1 = 0
    try:
        precision_0 = correct_0/ float(predict_0)
    except:
        precision_0 = 0
    try:
        recall_0 = correct_0/ float(true_0)
    except:
        recall_0 = 0
    try:
        Fmeasure_0 = 2*precision_0*recall_0 / float(precision_0 + recall_0)
    except:
        Fmeasure_0 = 0
    try:
        accuracy = (correct_0 + correct_1)/float(count)
    except:
        accuracy = 0

    # print ("sensitivity", recall_1)
    # print ("specificity", recall_0)
    try:
        # print ("harmonic", 2*recall_1*recall_0 / float(recall_1 + recall_0))
        return recall_1, recall_0, 2*recall_1*recall_0 / float(recall_1 + recall_0)
    except:
        return recall_1, recall_0, 0


def test():
    # for kernel_y, stride_y in zip([3, 2], [2, 1]):
    #     for kernel_x in [10, 15, 20, 30]:
    #         for stride_x in [3, 5]:
    #             for conv2_kernel_xs, conv2_stride_xs in zip([[5], [3]], [[3,2,1], [2,1]]):
    #                 for conv2_kernel_x in conv2_kernel_xs:
    #                     for conv2_stride_x in conv2_stride_xs:
    #                         for conv2_kernel_y in [2, 1]:
    #                             for conv3_kernel_x in [3, 2]:
    #                                 for conv3_kernel_y in [2, 1]:
    #                                     for epoch in range(1, 6):
    #                                         probfile = '_'.join(['readmitted_prob', str(kernel_y), str(kernel_x), str(stride_y), str(stride_x), str(conv2_kernel_x), str(conv2_kernel_y), str(conv2_stride_x), str(conv3_kernel_x), str(conv3_kernel_y), str(epoch)]) + ".csv"
    #                                         if not os.path.exists(probfile):
    #                                             print ('not exist file %s' % probfile)
    #                                             continue

    files = os.listdir('.')
    files = [f for f in files if f.endswith('.csv')]
    print ('#files = %d' % len(files))
    for probfile in files:
        # print ('parameter: ', str(kernel_y), str(kernel_x), str(stride_y), str(stride_x), str(epoch))
        probability = np.genfromtxt(probfile, delimiter=',', dtype=np.float32)
        prob, ytrue = probability[:, 0], probability[:, 1]
        if sum(np.isnan(prob)) > 0:
            print ('Input contains NaN, infinity or a value too large for dtype(\'float32\').')
            continue
        # print 'self calculate test accuracy = %f' % cal_accuracy(prob.reshape(-1, 1), ytrue.reshape(-1, 1))
        cnn_metric_dict = {} # for output to json
        cnn_metric_dict['Number of Samples: '] = ytrue.shape[0]
        cnn_sensitivity, cnn_specificity, cnn_harmonic = HealthcareMetrics(prob.reshape(-1, 1), ytrue.reshape(-1, 1), 0.25)
        cnn_metric_dict['AUC: '] = auroc(prob.reshape(-1, 1), ytrue.reshape(-1, 1))
        cnn_metric_dict['accuracy: '] = cal_accuracy(prob.reshape(-1, 1), ytrue.reshape(-1, 1))
        cnn_metric_dict['Sensitivity: '] = cnn_sensitivity
        cnn_metric_dict['Specificity: '] = cnn_specificity

        try:
            with open('model_comparison.json', 'a') as cnn_metric_info_writer:
                cnn_metric_info_writer.write('[')
                cnn_metric_info_writer.write('\"prob file: %s\", ' % (probfile))
                cnn_metric_info_writer.write('\"AUC: %s\", '% (str(int(100 * round((auroc(prob.reshape(-1, 1), ytrue.reshape(-1, 1))),2)))+'%') )
                cnn_metric_info_writer.write('\"Accuracy: %s\", ' % (str(int(100 * cnn_metric_dict['accuracy: '])) + '%'))
                cnn_metric_info_writer.write('\"Sensitivity: %s\", '%(str(int(100 * round(cnn_sensitivity,2)))+'%'))
                cnn_metric_info_writer.write('\"Specificity: %s\" '%(str(int(100* round(cnn_specificity,2)))+'%'))
                cnn_metric_info_writer.write(']\n')
        except Exception as e:
            os.remove('model_comparison.json')
            print('output cnn_metric_info.json failed: ', e)
    

test()
