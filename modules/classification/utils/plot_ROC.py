"""
=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

Multiclass settings
-------------------

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-class classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`.

"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def plot_ROC(y_score, y_test, n_iter, epoch, save_dir='', mode='test', debug=False):
    """
    ROC curve
    :param y_score: (N, num_classes) predicted values (e.g., softmax(logits)-1)
    :param y_test:  (N, num_classes) one_hot GT vector
    :param save_dir: Where to save the ROC thresholds file, if desired
    :param mode:     Tag for saving (e.g., "train" or "test")
    :return:         None
    """
    n_classes = y_test.shape[1]

    # Dictionaries to hold per-class metrics
    fpr = {}
    tpr = {}
    thresholds_dict = {}
    roc_auc = {}

    # Compute ROC curve for each class that has positive samples
    for i in range(n_classes):
        num_pos = np.sum(y_test[:, i])
        if num_pos == 0:
            # If no positive samples exist for class i, skip it
            print(f"Skipping class {i} (no positive samples).")
            continue

        fpr[i], tpr[i], thresholds_dict[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Optionally save threshold info for the last class, if it exists
    last_class = n_classes - 1
    if last_class in tpr:
        if save_dir != '':
            ROC_path = Path(save_dir) / f"ROC_{mode}.txt"
            with open(ROC_path, 'a+') as f:
                f.write('k\tthresh\t\tTPR\t\tFPR')
                for k in range(len(tpr[last_class])):
                    # thresholds_dict[last_class][k] + 1 is used per your original code
                    f.write('\n{}\t{:.5f}\t{:.2f}\t\t{:.2f}'.format(
                        k,
                        thresholds_dict[last_class][k] + 1,
                        100 * tpr[last_class][k],
                        100 * fpr[last_class][k]
                    ))
    else:
        print(f"Class {last_class} does not exist (it was skipped or not in y_test).")

    # Compute micro-average ROC
    # (aggregates the contributions of all classes)
    fpr["micro"], tpr["micro"], thr = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    last_class = n_classes - 1
    if last_class in fpr:
        plt.figure()
        lw = 2
        plt.plot(fpr[last_class], tpr[last_class], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[last_class])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        # Plot ROC curves for the multiclass problem
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        if debug:
            plt.show()

    else:
        print(f"Class {last_class} was skipped, so no ROC plot for it.")

    if save_dir != '':
        plot_dir = Path(save_dir) / 'plots'
        if not plot_dir.exists():
            plot_dir.mkdir(parents=True)
        ROC_path = Path(plot_dir)/f"ROC_{mode}_{epoch}_{n_iter}.png"
        plt.savefig(ROC_path)



if __name__ == '__main__':
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                     random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    plot_ROC(y_score, y_test)