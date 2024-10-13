# -*- coding: utf-8 -*-
"""
Module for evaluating the performance of a trained model using various classification metrics.
"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    """
    The Evaluator class provides methods to compute and display various
    classification metrics for a trained model.
    """

    def __init__(self):
        """
        Initializes the Evaluator class.
        """

    def evaluate(self, y_true, y_pred, y_pred_prob):
        """
        Evaluates the model using various classification metrics and prints the results.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            y_pred_prob (array-like): Predicted probabilities for the positive class.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
        }
