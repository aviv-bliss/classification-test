import numpy as np

from metrics.metric_builder import Metric
from utils.auxiliary import softmax
from sklearn.metrics import confusion_matrix


class accuracy_onehot(Metric):
    def __init__(self, x, y, FLAGS):
        super(accuracy_onehot, self).__init__(x, y, FLAGS)
        self.logits = x
        self.y = y

    def build(self):
        correct_pred = np.sum(np.equal(np.argmax(self.logits, 1), np.argmax(self.y, 1)))
        acc = correct_pred * 100.0 / len(self.y)

        return acc


class accuracy_categ(Metric):
    def __init__(self, x, y, FLAGS):
        super(accuracy_categ, self).__init__(x, y, FLAGS)
        self.logits = x
        self.y = y

    def build(self):
        correct_pred = np.sum(np.equal(np.argmax(self.logits, 1), self.y))
        acc = correct_pred * 100.0 / len(self.y)
        return acc


def softmax(logit_row):
    """
    A simple row-wise softmax.
    E.g., if logit_row = [z1, z2, ..., zC], returns an array of shape (C,).
    """
    exps = np.exp(logit_row - np.max(logit_row))
    return exps / np.sum(exps)


class F1(Metric):
    def __init__(self, x, y, FLAGS):
        super(F1, self).__init__(x, y, FLAGS)
        # x is assumed to be a list or iterable of logits with shape [N, num_classes]
        # Convert logits to probabilities
        probability = np.array([softmax(x[i]) for i in range(len(x))])  # shape: (N, num_classes)

        # Multi-class prediction: take argmax across classes
        self.pred = np.argmax(probability, axis=1)  # shape: (N, )
        self.y = y                                  # true class labels, shape: (N, )
        self.n_classes = probability.shape[1]

    def build(self):
        """
        Computes a macro-averaged F1 score across all classes.
        """
        # Compute confusion matrix of shape [n_classes, n_classes]
        cm = confusion_matrix(self.y, self.pred, labels=range(self.n_classes))

        class_f1s = []
        for c in range(self.n_classes):
            # True Positives for class c
            TP = cm[c, c]
            # False Positives: sum of column c minus TP
            FP = cm[:, c].sum() - TP
            # False Negatives: sum of row c minus TP
            FN = cm[c, :].sum() - TP

            # precision = TP/(TP+FP)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            # recall = TP/(TP+FN)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            # F1 for class c
            f1_c = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            class_f1s.append(f1_c)

        # Macro-average: mean of per-class F1
        macro_f1 = np.mean(class_f1s)
        return macro_f1
