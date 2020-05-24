"""Metrics for evaluation"""
# Copyright (C) 2020 Amir Alansary <amiralansary@gmail.com>
#

import torch


###############################################################################
# Classification Accuracy
###############################################################################
def accuracy(output, target, topk=(1,)):
    """Computes the classification accuracy over the k top predictions for the
    specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


###############################################################################
# Confusion Matrix
###############################################################################
class EvalMetric(object):
    """A class contains different metrics that can be used for evaluation"""

    def __init__(self, target, pred, num_classes, verbose=False):
        """
        Args:
            target: target labels
            pred: model predictions
            num_classes (int): number of classes
            verbose (bool): display detailed results (Defualt=False)
        """
        super().__init__()
        self.target = target
        self.pred = pred
        self.num_classes = num_classes
        self.verbose = verbose
        self.conf_matrix = self.ConfusionMatrix
        self.true_false_pos_neg = self.TrueFalseCondition
        if self.verbose:
            print('Number of classes {} - Size of target labels {} - '
                  'Size of predicted labels = {}'.format(self.num_classes,
                                                         self.target.size,
                                                         self.pred.size))

    # =========================================================================
    # Confusion Matrix
    @property
    def ConfusionMatrix(self):
        """Computes the confusion matrix"""
        conf_matrix = torch.zeros([self.num_classes, self.num_classes], dtype=torch.int32)
        for t, p in zip(self.target, self.pred):
            conf_matrix[t, p] += 1
        if self.verbose:
            print('Confusion matrix\n', conf_matrix)

        return conf_matrix

    # =========================================================================
    # True/False, Postive/Negative
    @property
    def TrueFalseCondition(self):
        """Computes true VS predicted conditon values:
        Retrurns:
            [True Positive (TP), True Negative (TN),
            False Positive (FP), False Negative (FN)]
        """
        conf_matrix = self.conf_matrix
        TP = conf_matrix.diag()
        TN = 0 * TP
        FP = 0 * TP
        FN = 0 * TP
        for c in range(self.num_classes):
            idx = torch.ones(self.num_classes).byte()
            idx[c] = 0
            # all non-class samples classified as non-class
            TN[c] = conf_matrix[idx.nonzero()[:,
                                None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
            # all non-class samples classified as class
            FP[c] = conf_matrix[idx, c].sum()
            # all class samples not classified as class
            FN[c] = conf_matrix[c, idx].sum()
            if self.verbose:
                print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
                    c, TP[c], TN[c], FP[c], FN[c]))

        return TP, TN, FP, FN

    def ClassificationAccuracy(self):
        """Compute classification rate or accuracy:
            (TP+TN)/(TP+TN+FP+FN)
        """
        TFPN = self.true_false_pos_neg
        return (TFPN[0] + TFPN[1]) / TFPN.sum()

    def precision(self):
        """Compute precision:
            TP / (TP + FP)
        """
        TFPN = self.true_false_pos_neg
        return TFPN[0] / (TFPN[0]+TFPN[2])

    def sensitivity(self):
        """Compute sensitivity (recall):
           TP / (TP + FN)
        """
        TFPN = self.true_false_pos_neg
        return TFPN[0] / (TFPN[0]+TFPN[3])

    def f1score(self):
        """Compute sensitivity (recall):
            (2 * (Precision * Recall))/(Precision + Recall)
        """
        return 2.0 * (self.precision * self.sensitivity)/(self.precision + self.sensitivity)


###############################################################################
# Sensitivity (TPR)
###############################################################################
# TP/ (FN+TP)

###############################################################################
# Specificity (FPR)
###############################################################################
# FP / (FP+TN)


###############################################################################
# Area Under Curve (AUC) - Receiver Operating Characteristic (ROC)
###############################################################################


###############################################################################
# F1 Score
###############################################################################


#