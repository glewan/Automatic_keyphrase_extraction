import numpy as np
import statistics


def evaluate_document(test, predictions, verbose):
    #print(test)
    #print(predictions)
    #print('')
    predicted_len = len(predictions)
    test_len = len(test)
    hits = 0
    AP = 0
    precision_5 = 0
    test = [p for (p, s) in test]
    for index, phrase in enumerate(predictions):
        if phrase[0] in test:
            hits += 1
            AP += hits/(index+1)
            if index <= 4:
                precision_5 += 1

    precision = hits/predicted_len
    recall = hits/test_len
    f1 = 2*(precision * recall)/(precision + recall) if precision + recall > 0 else 0
    AP /= test_len
    precision_5 /= 5
    if verbose:
        print("Precision: {:.4f},\nRecall: {:.4f},\nF1-measure: {:.4f},\nAP: {:.4f},\nP@5: {:.4f}"
            .format(precision, recall, f1, AP, precision_5))
        print("--------------")
    return precision, recall, f1, AP, precision_5


def evaluate_dataset(test_set, predictions_set, verbose):

    test_set_len = len(test_set)
    dataset_metrics = np.empty((test_set_len, 5))

    for index in range(test_set_len):
        dataset_metrics[index] = evaluate_document(test_set[index], predictions_set[index], verbose) 

    av_precision = statistics.mean(dataset_metrics[:, 0])
    av_recall = statistics.mean(dataset_metrics[:, 1])
    av_f1 = statistics.mean(dataset_metrics[:, 2])
    MAP = statistics.mean(dataset_metrics[:, 3])
    av_precision_5 = statistics.mean(dataset_metrics[:, 4])

    print("Average precision: {:.4f},\nAverage recall: {:.4f},\nAverage F1-measure: {:.4f},\nMAP: {:.4f},"
          "\nAverage precision@5: {:.4f}"
          .format(av_precision, av_recall, av_f1, MAP, av_precision_5))
    print("--------------")
    return av_precision, av_recall, av_f1, MAP, av_precision_5
