# (1) tuning threshold
# (2) [100, 248], 148 samples in all
import itertools
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import sys

def HealthcareMetrics(y_prob, y_true, prob_threshold):
    # file = open(y_prob)
    # y_scores = np.genfromtxt(file, delimiter=",")
    y_scores = y_prob[0:351]
    # print "y_scores shape = \n", y_scores.shape

    # file = open(y_true)
    # y_true = np.genfromtxt(file, delimiter=",")
    y_true = y_true[0:351]
    y_true = y_true.astype(np.int)
    # print "y_true shape = \n", y_true.shape

    threshold = prob_threshold
    # print "threshold: ", threshold
    count = y_scores.shape[0]
    true_0 = 0
    true_1 = 0
    predict_0 = 0
    predict_1 = 0
    correct_0 = 0
    correct_1 = 0
    for i in range(0, count):
        #print "y_scores"
        #print y_scores[i]
        if y_true[i] == 1:
            true_1 = true_1 + 1
        else:
            true_0 = true_0 + 1

        if y_scores[i] >= threshold:
            predict_1 = predict_1 + 1
        else:
            predict_0 = predict_0 + 1

        if y_true[i] == 1 and y_scores[i] >= threshold:
            correct_1 = correct_1 + 1
        if y_true[i] == 0 and y_scores[i] < threshold:
            correct_0 = correct_0 + 1

   
    print "Metric Begin"

    #print "count"
    #print count

    precision_1 = correct_1 / float(predict_1)
    #print "precision_1"
    #print precision_1
    recall_1 = correct_1 / float(true_1)
    #print "recall_1"
    #print recall_1

    Fmeasure_1 = 2*precision_1*recall_1 / float(precision_1 + recall_1)
    #print "F-measure_1"
    #print Fmeasure_1

    precision_0 = correct_0/ float(predict_0)
    #print "precision_0"
    #print precision_0
    recall_0 = correct_0/ float(true_0)
    #print "recall_0"
    #print recall_0
    Fmeasure_0 = 2*precision_0*recall_0 / float(precision_0 + recall_0)
    #print "F_measure_0"
    #print Fmeasure_0

    accuracy = (correct_0 + correct_1)/float(count)
    #print "overall accuracy"
    #print accuracy

    # python scores

    #print "python accuracy"
    # print "y_true: ", y_true
    # print "y_scores: ", y_scores
    #print accuracy_score(y_true, (y_scores>threshold).astype(int))

    #print "python precision"
    #print precision_score(y_true, (y_scores>threshold).astype(int))

    #print "python recall"
    #print recall_score(y_true, (y_scores>threshold).astype(int))

    #print "python auc"
    #print roc_auc_score(y_true, y_scores)

    print "sensitivity"
    print recall_1
    print "specificity"
    print recall_0
    print "harmonic"
    print 2*recall_1*recall_0 / float(recall_1 + recall_0)
    return recall_1, recall_0, 2*recall_1*recall_0 / float(recall_1 + recall_0)
# python test_metrics_threshold.py 0.115
# python test_metrics_threshold.py 0.209 (all)
# python test_metrics_threshold.py 0.25 (all)
