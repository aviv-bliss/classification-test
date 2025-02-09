import numpy as np
import itertools
import os
import numpy as np

import dataloaders.img_transforms as transforms
from dataloaders.dataloader_builder import DataLoader
from metrics.metric_builder import Metric
from tests.test_builder import Test
from utils.auxiliary import AverageMeter, save_loss_to_resultstable, check_if_best_model_and_save, \
    load_model_and_weights, softmax
from utils.plot_ROC import plot_ROC
from sklearn.metrics import confusion_matrix


def softmax(logit_row):
    exps = np.exp(logit_row - np.max(logit_row))
    return exps / np.sum(exps)


def evaluate_nclass_metrics(x, y, train_dir, n_iter, epoch, mode, debug):
    """
    Evaluates classification metrics for an n-class problem using logits from inputs x
    and integer labels in y. Maintains structure similar to original binary version,
    but adapted for multi-class.
    Args:
        self: Object containing debug flags, directories, etc.
        x:    List or iterable of logit arrays ( shape: [N, n] ) - predictions
        y:    List or iterable of integer labels ( shape: [N,] ), GT, each in {0,1,...,n-1}
    Returns:
        None (prints out various metrics, confusion matrix, etc.)
    """
    # Flatten out lists of arrays, if x and y come in chunks
    logits = np.array(list(itertools.chain(*x)))  # shape [N, n]
    num_classes = logits.shape[1]
    y = np.array(list(itertools.chain(*y)))  # shape [N,   ]

    # One-hot for potential usage in ROC or any other multi-class metric
    y = y.astype(np.int32)
    y_onehot = (np.eye(logits.shape[1])[y]).astype(np.int32)  # shape [N, n]


    # Convert logits to probabilities
    probability = np.array([softmax(logits[i]) for i in range(len(logits))])  # shape [N, n]
    pred = np.argmax(probability, axis=1)

    # In debug mode, plot ROC (multi-class version). Make sure your plot_ROC can handle n-class data.

    plot_ROC(probability, y_onehot, n_iter, epoch, train_dir, mode, debug)

    # Compute confusion matrix of shape [n x n]
    cm = confusion_matrix(y, pred, labels=range(num_classes))

    # Overall accuracy
    total_samples = len(y)
    correct_pred = np.sum(pred == y)
    acc = correct_pred / total_samples if total_samples > 0 else 0.0

    print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
    print(f"\nN (total samples): {total_samples}")
    print(f"Correct predictions: {correct_pred} ({acc * 100:.2f}%)")

    # Per-class precision, recall, and F1 using confusion matrix
    class_precisions = []
    class_recalls = []
    class_f1s = []

    for i in range(num_classes):
        # True positives for class i = cm[i, i]
        TP = cm[i, i]
        # False positives for class i = sum of column i minus TP
        FP = cm[:, i].sum() - TP
        # False negatives for class i = sum of row i minus TP
        FN = cm[i, :].sum() - TP
        # True negatives for class i = total samples minus (TP+FP+FN)
        TN = cm.sum() - (TP + FP + FN)

        precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_i = (2 * precision_i * recall_i / (precision_i + recall_i)
                if (precision_i + recall_i) > 0 else 0.0)

        class_precisions.append(precision_i)
        class_recalls.append(recall_i)
        class_f1s.append(f1_i)

        if debug:
            print(f"\nClass {i}:")
            print(f"  TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
            print(f"  Precision: {precision_i:.4f}, Recall: {recall_i:.4f}, F1: {f1_i:.4f}")

    # Macro-average metrics
    macro_precision = np.mean(class_precisions)
    macro_recall = np.mean(class_recalls)
    macro_f1 = np.mean(class_f1s)

    print(f"\nOverall Accuracy (acc): {acc * 100:.2f}%")
    print(f"Macro Precision:        {macro_precision * 100:.2f}%")
    print(f"Macro Recall:           {macro_recall * 100:.2f}%")
    print(f"Macro F1:               {macro_f1 * 100:.2f}%")

    # save results to results_table.csv
    col_names = ['acc', 'macro_precision', 'macro_recall', 'macro_f1']
    values = [acc, macro_precision, macro_recall, macro_f1]
    results_table_path = os.path.join(train_dir, 'results_table.csv')
    save_loss_to_resultstable(values, col_names, results_table_path, n_iter, epoch)


    return logits, acc, macro_precision, macro_recall, macro_f1









