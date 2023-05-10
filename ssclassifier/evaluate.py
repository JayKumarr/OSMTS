import numpy as np
import sklearn
from sklearn.metrics import hamming_loss


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def precision_score(y_true, y_pred):
    sum_of_all_precision = 0
    number_of_instances = y_true.shape[0]
    for i in range(number_of_instances):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        intersection = list(set_true & set_pred)
        precision_of_single_instance=intersection.__len__() / set_pred.__len__()
        sum_of_all_precision+=precision_of_single_instance

    precision = sum_of_all_precision/number_of_instances
    return precision


def recall_score(y_true, y_pred):
    sum_of_all_precision = 0
    number_of_instances = y_true.shape[0]
    for i in range(number_of_instances):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        intersection = list(set_true & set_pred)
        precision_of_single_instance=intersection.__len__() / set_true.__len__()
        sum_of_all_precision+=precision_of_single_instance

    precision = sum_of_all_precision/number_of_instances
    return precision


def print_evaluation_score(true_labels, predicted_labels):
    headers = " Hamming loss: Hamming Score: Accuracy: Precision: Average Micro Precision: Average Macro Precision: Recall: Micro F1 score: Macro F1 score: Micro Jaccard score: Macro Jaccard score: Macro RoC AUC score: Micro RoC AUC score: "
    common_string = ' {0}'.format(hamming_loss(true_labels, predicted_labels))
    common_string = common_string + (', {0}'.format(hamming_score(true_labels, predicted_labels)))
    common_string = common_string + (', {0}'.format(sklearn.metrics.accuracy_score(true_labels, predicted_labels, normalize=True, sample_weight=None)))
    common_string = common_string + (', {0}'.format(precision_score(true_labels, predicted_labels)))
    common_string = common_string + (', {0}'.format(sklearn.metrics.average_precision_score(true_labels, predicted_labels, average="micro")))
    common_string = common_string + (', {0}'.format(sklearn.metrics.average_precision_score(true_labels, predicted_labels, average="macro")))
    common_string = common_string + (', {0}'.format(recall_score(true_labels, predicted_labels)))
    common_string = common_string + (", {0}".format( sklearn.metrics.f1_score(true_labels, predicted_labels, average="micro")))
    common_string = common_string + (", {0}".format(sklearn.metrics.f1_score(true_labels, predicted_labels, average="macro")))
    common_string = common_string + (", {0}".format(sklearn.metrics.f1_score(true_labels, predicted_labels, average="micro")))
    common_string = common_string + (", {0}".format(sklearn.metrics.f1_score(true_labels, predicted_labels, average="macro")))
    common_string = common_string + (", {0}".format(sklearn.metrics.roc_auc_score(true_labels, predicted_labels, average="macro")))
    common_string = common_string + (", {0}".format(sklearn.metrics.roc_auc_score(true_labels, predicted_labels, average="micro")))



    return headers.replace(":","\t"),common_string.replace(",","\t")

if __name__ == '__main__':

    true_labels = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]])

    predicted_labels = np.array([[1, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]])

    print_evaluation_score(true_labels,predicted_labels)