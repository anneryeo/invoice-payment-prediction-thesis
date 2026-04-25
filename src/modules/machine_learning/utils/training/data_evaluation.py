from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import numpy as np


def data_evaluation(y_pred, y_test, y_proba=None):
    """
    Evaluate classification model performance and generate chart-ready curve data.

    Supports both binary and multiclass classification. When multiclass is
    detected (more than 2 unique classes in y_test), ROC and PR curves are
    computed per class using a One-vs-Rest (OvR) strategy: each class is
    treated as the positive class and all remaining classes as negative.
    This is consistent with how roc_auc_score is computed (multi_class="ovr").

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    y_test : array-like of shape (n_samples,)
        True class labels.
    y_proba : array-like of shape (n_samples, n_classes) or (n_samples,), optional
        Predicted probabilities. If 2D, each column corresponds to a class
        in sorted order. Required for AUC and curve data. If None, only
        core metrics and confusion matrix are returned.

    Returns
    -------
    dict
        A flat dictionary with the following keys:

        - accuracy           : float
        - precision_macro    : float
        - recall_macro       : float
        - f1_macro           : float
        - roc_auc_macro      : float or None
              Macro-averaged AUC (OvR). None if y_proba is not provided
              or if computation fails (e.g. missing classes in y_test).
        - confusion_matrix   : list of list of int
              2D matrix of shape (n_classes, n_classes). Rows are true
              labels, columns are predicted labels.
        - roc_curve          : dict or None
              Binary — single curve:
                  {"fpr": [...], "tpr": [...]}
              Multiclass — one curve per class (OvR), keyed by str(class):
                  {"0": {"fpr": [...], "tpr": [...]}, ...}
        - pr_curve           : dict or None
              Binary — single curve:
                  {"precision": [...], "recall": [...]}
              Multiclass — one curve per class (OvR), keyed by str(class):
                  {"0": {"precision": [...], "recall": [...]}, ...}

              Note: roc_curve and pr_curve are None if y_proba is not
              provided or if AUC computation fails.

    Examples
    --------
    Binary classification:

        result = data_evaluation(y_pred, y_test, y_proba=model.predict_proba(X_test))
        result["accuracy"]               # float
        result["roc_auc_macro"]          # float
        result["roc_curve"]["fpr"]       # list
        result["confusion_matrix"]       # 2D list

    Multiclass classification (e.g. 3 classes):

        result = data_evaluation(y_pred, y_test, y_proba=model.predict_proba(X_test))
        result["roc_auc_macro"]          # float, macro OvR AUC
        result["roc_curve"]["0"]["fpr"]  # list, class 0 vs rest
        result["pr_curve"]["2"]["recall"]# list, class 2 vs rest
    """
    classes       = sorted(np.unique(y_test))
    is_multiclass = len(classes) > 2

    result = {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision_macro":  precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro":         f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_macro":    None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve":        None,
        "pr_curve":         None,
    }

    if y_proba is not None:
        try:
            result["roc_auc_macro"] = roc_auc_score(
                y_test, y_proba, average="macro", multi_class="ovr"
            )

            if is_multiclass:
                y_test_binarized = label_binarize(y_test, classes=classes)
                roc_curves, pr_curves = {}, {}

                for i, cls in enumerate(classes):
                    fpr, tpr, _  = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                    prec, rec, _ = precision_recall_curve(y_test_binarized[:, i], y_proba[:, i])
                    roc_curves[str(cls)] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                    pr_curves[str(cls)]  = {"precision": prec.tolist(), "recall": rec.tolist()}

                result["roc_curve"] = roc_curves
                result["pr_curve"]  = pr_curves

            else:
                y_score      = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                fpr, tpr, _  = roc_curve(y_test, y_score)
                prec, rec, _ = precision_recall_curve(y_test, y_score)

                result["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                result["pr_curve"]  = {"precision": prec.tolist(), "recall": rec.tolist()}

        except ValueError:
            pass

    return result